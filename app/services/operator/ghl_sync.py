"""GHL Sync Service — pulls GHL data for orphan leads and syncs to Neo4j.

Pulls contact details, conversation messages, notes, and tags from GHL,
matches tags against Building nodes in Neo4j (fuzzy), and persists
everything in a single Neo4j transaction.

Called by the lead_operator beat task with 5s delays between leads,
so no internal rate limiting is needed. GHL calls are made sequentially
(not in parallel) to respect rate limits.
"""

from __future__ import annotations

import hashlib

import structlog
from neo4j import AsyncDriver

from app.services import ghl_service

logger = structlog.get_logger()


class GHLSyncService:
    """Syncs a single orphan lead's GHL data into Neo4j."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def sync_lead(self, contact_id: str, phone: str) -> dict:
        """Pull GHL data for a single lead and sync to Neo4j.

        Returns dict with: contact, message_count, notes_count,
        buildings_matched, messages_summary.

        Raises on get_contact failure (can't proceed without contact data).
        Fails open on conversations/messages/notes (non-critical).
        """
        # 1. Get contact details (REQUIRED — raise on failure)
        contact_response = await ghl_service.get_contact(contact_id)
        contact = contact_response.get("contact", contact_response)

        name = f"{contact.get('firstName', '')} {contact.get('lastName', '')}".strip()
        email = str(contact.get("email", "") or "")
        contact_phone = str(contact.get("phone", phone) or phone or "")
        tags = contact.get("tags", []) or []
        date_added = str(contact.get("dateAdded", "") or "")
        source = str(contact.get("source", "") or "")

        # 2. Search for conversation and get messages (fail-open)
        messages: list[dict] = []
        try:
            conversation = await ghl_service.search_conversations(contact_id)
            if conversation:
                conversation_id = conversation.get("id", "")
                if conversation_id:
                    messages = await ghl_service.get_conversation_messages(
                        conversation_id, limit=20
                    )
        except Exception as e:
            logger.warning(
                "operator.sync_conversations_failed",
                contact_id=contact_id,
                error=str(e),
            )

        # 3. Get notes (fail-open)
        notes: list[dict] = []
        try:
            notes = await ghl_service.get_contact_notes(contact_id)
        except Exception as e:
            logger.warning(
                "operator.sync_notes_failed",
                contact_id=contact_id,
                error=str(e),
            )

        # 4. Match tags against Building nodes in Neo4j
        buildings_matched: list[dict] = []
        for tag in tags:
            if not tag or not isinstance(tag, str):
                continue
            matched = await self._match_building_by_tag(tag)
            if matched:
                buildings_matched.extend(matched)

        # Deduplicate matched buildings by building_id
        seen_ids: set[str] = set()
        unique_buildings: list[dict] = []
        for b in buildings_matched:
            bid = b.get("building_id", "")
            if bid and bid not in seen_ids:
                seen_ids.add(bid)
                unique_buildings.append(b)
        buildings_matched = unique_buildings

        # 5. Sync everything to Neo4j in a single transaction
        notes_text = "\n---\n".join(
            n.get("body", "") for n in notes if isinstance(n, dict)
        )

        await self._sync_to_neo4j(
            contact_id=contact_id,
            name=name,
            email=email,
            phone=contact_phone,
            source=source,
            date_added=date_added,
            notes_text=notes_text,
            messages=messages,
            buildings=buildings_matched,
        )

        # 6. Build messages summary (last 5 messages)
        messages_summary: list[dict] = []
        for msg in messages[:5]:
            direction = msg.get("direction", "")
            role = "user" if direction == "inbound" else "assistant"
            content = msg.get("body", msg.get("message", ""))
            messages_summary.append({"role": role, "content": content[:200]})

        logger.info(
            "operator.sync_complete",
            contact_id=contact_id,
            name=name,
            message_count=len(messages),
            notes_count=len(notes),
            buildings_matched=[b["name"] for b in buildings_matched],
        )

        return {
            "contact": {"name": name, "email": email, "tags": tags},
            "message_count": len(messages),
            "notes_count": len(notes),
            "buildings_matched": [b["name"] for b in buildings_matched],
            "messages_summary": messages_summary,
        }

    async def _match_building_by_tag(self, tag: str) -> list[dict]:
        """Match a GHL tag against Building nodes using case-insensitive CONTAINS."""
        async with self._driver.session() as session:
            result = await session.execute_read(self._match_building_tx, tag)
            return result

    @staticmethod
    async def _match_building_tx(tx, tag: str) -> list[dict]:
        result = await tx.run(
            """
            MATCH (b:Building)
            WHERE toLower(b.name) CONTAINS toLower($tag)
            RETURN b.name AS name, b.building_id AS building_id
            """,
            tag=tag,
        )
        return [dict(record) async for record in result]

    async def _sync_to_neo4j(
        self,
        contact_id: str,
        name: str,
        email: str,
        phone: str,
        source: str,
        date_added: str,
        notes_text: str,
        messages: list[dict],
        buildings: list[dict],
    ) -> None:
        """Persist all synced data to Neo4j in a single write transaction."""
        async with self._driver.session() as session:
            await session.execute_write(
                self._sync_tx,
                contact_id,
                name,
                email,
                phone,
                source,
                date_added,
                notes_text,
                messages,
                buildings,
            )

    @staticmethod
    async def _sync_tx(
        tx,
        contact_id: str,
        name: str,
        email: str,
        phone: str,
        source: str,
        date_added: str,
        notes_text: str,
        messages: list[dict],
        buildings: list[dict],
    ) -> None:
        # MERGE Lead node
        await tx.run(
            """
            MERGE (l:Lead {ghl_contact_id: $cid})
            SET l.name = $name,
                l.email = $email,
                l.phone = $phone,
                l.source = $source,
                l.ghl_date_added = $date_added,
                l.ghl_notes = $notes_text,
                l.synced_by_operator = true,
                l.operator_synced_at = datetime(),
                l.updatedAt = datetime()
            """,
            cid=contact_id,
            name=name,
            email=email,
            phone=phone,
            source=source,
            date_added=date_added,
            notes_text=notes_text,
        )

        # Create Interaction nodes for each GHL message
        for msg in messages:
            direction = msg.get("direction", "")
            role = "user" if direction == "inbound" else "assistant"
            content = msg.get("body", msg.get("message", ""))
            channel = str(msg.get("type", "sms") or "sms").lower()
            created_at = msg.get("dateAdded", "")

            # Build unique key: ghl_message_id if available, else content+created_at hash
            ghl_msg_id = msg.get("id", msg.get("messageId", ""))
            if ghl_msg_id:
                merge_key = ghl_msg_id
            else:
                raw = f"{content}{created_at}"
                merge_key = hashlib.sha256(raw.encode()).hexdigest()[:16]

            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})
                MERGE (i:Interaction {merge_key: $merge_key})
                ON CREATE SET
                    i.role = $role,
                    i.content = $content,
                    i.channel = $channel,
                    i.created_at = $created_at,
                    i.source = 'ghl_operator_sync'
                MERGE (l)-[:HAS_INTERACTION]->(i)
                """,
                cid=contact_id,
                merge_key=merge_key,
                role=role,
                content=content,
                channel=channel,
                created_at=created_at,
            )

        # MERGE INTERESTED_IN relationships to matched buildings
        for building in buildings:
            await tx.run(
                """
                MATCH (l:Lead {ghl_contact_id: $cid})
                MATCH (b:Building {building_id: $bid})
                MERGE (l)-[:INTERESTED_IN]->(b)
                """,
                cid=contact_id,
                bid=building["building_id"],
            )
