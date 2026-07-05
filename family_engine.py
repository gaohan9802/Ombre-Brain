# ============================================================
# Module: Family Engine (family_engine.py)
# 模块：家族聚合引擎
#
# Layer 2 of the memory system: automatic clustering of related
# memories into "families" with auto-generated narrative summaries.
# 记忆系统第二层：将相关记忆自动聚类为"家族"，并生成叙事摘要。
#
# Core concepts:
#   - Family: a cluster of semantically related memory buckets
#   - Centroid: average embedding vector of family members
#   - Summary: AI-generated narrative summary (200-300 chars)
#   - Emotion trend: valence/arousal trajectory over time
#   - Lines: cross-cutting verb/topic dimensions for cross-family queries
#
# Triggered by:
#   - hold: new memory → try assign to existing family or create new
#   - breath: surface family summaries instead of individual buckets
# ============================================================

import os
import json
import math
import logging
import time
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("ombre_brain.family")

# --- Constants ---
DEFAULT_CLUSTER_THRESHOLD = 0.72  # cosine similarity threshold for family assignment
MIN_FAMILY_SIZE_FOR_SUMMARY = 5   # minimum members before generating summary
MAX_SUMMARY_LENGTH = 300          # max chars for family summary
ORPHAN_CLUSTER_THRESHOLD = 0.70   # threshold for orphan-to-orphan clustering
MIN_ORPHANS_FOR_NEW_FAMILY = 2    # minimum orphans to form a new family


class FamilyEngine:
    """
    Memory family clustering engine.
    Manages family metadata, clustering, and summary generation.
    """

    def __init__(self, config: dict, embedding_engine, dehydrator, bucket_mgr):
        self.config = config
        self.embedding_engine = embedding_engine
        self.dehydrator = dehydrator
        self.bucket_mgr = bucket_mgr

        # Storage path
        self.base_dir = config.get("buckets_dir", "./buckets")
        self.families_path = os.path.join(self.base_dir, "families.json")

        # Configurable threshold
        family_cfg = config.get("family", {}) or {}
        self.cluster_threshold = float(family_cfg.get("cluster_threshold", DEFAULT_CLUSTER_THRESHOLD))
        self.min_summary_size = int(family_cfg.get("min_summary_size", MIN_FAMILY_SIZE_FOR_SUMMARY))

        # Load existing families
        self.families = self._load_families()

    # ============================================================
    # Persistence
    # ============================================================

    def _load_families(self) -> dict:
        """Load families from JSON file."""
        if os.path.exists(self.families_path):
            try:
                with open(self.families_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"Failed to load families.json: {e}")
        return {}

    def _save_families(self):
        """Persist families to JSON file."""
        os.makedirs(os.path.dirname(self.families_path), exist_ok=True)
        tmp = self.families_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.families, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.families_path)
        except Exception as e:
            logger.warning(f"Failed to save families.json: {e}")

    # ============================================================
    # Core: Assign bucket to family on hold
    # ============================================================

    async def assign_bucket(self, bucket_id: str, content: str, metadata: dict) -> Optional[str]:
        """
        Try to assign a new bucket to an existing family.
        Returns family_id if assigned, None if orphan.
        
        Called after hold creates/merges a bucket.
        """
        if not self.embedding_engine or not self.embedding_engine.enabled:
            logger.info("Embedding engine not available, skipping family assignment")
            return None

        # Get the bucket's embedding
        bucket_emb = await self.embedding_engine.get_embedding(bucket_id)
        if bucket_emb is None:
            try:
                await self.embedding_engine.generate_and_store(bucket_id, content)
                bucket_emb = await self.embedding_engine.get_embedding(bucket_id)
            except Exception:
                pass
        if bucket_emb is None:
            return None

        # Find best matching family by centroid similarity
        best_family_id = None
        best_sim = 0.0

        for fam_id, fam in self.families.items():
            centroid = fam.get("centroid")
            if not centroid:
                continue
            sim = self._cosine_similarity(bucket_emb, centroid)
            if sim > best_sim:
                best_sim = sim
                best_family_id = fam_id

        if best_family_id and best_sim >= self.cluster_threshold:
            # Assign to existing family
            await self._add_to_family(best_family_id, bucket_id, bucket_emb, metadata)
            logger.info(f"Bucket {bucket_id} assigned to family {best_family_id} (sim={best_sim:.3f})")
            return best_family_id

        # Not assigned — try to form a new family from orphans
        new_family_id = await self._try_form_family_from_orphans(bucket_id, bucket_emb, metadata)
        if new_family_id:
            logger.info(f"New family {new_family_id} formed with bucket {bucket_id}")
            return new_family_id

        logger.info(f"Bucket {bucket_id} remains orphan (best_sim={best_sim:.3f})")
        return None

    async def _add_to_family(self, family_id: str, bucket_id: str, bucket_emb: list, metadata: dict):
        """Add a bucket to an existing family and update centroid."""
        fam = self.families[family_id]
        members = fam.get("members", [])
        if bucket_id not in members:
            members.append(bucket_id)
            fam["members"] = members

        # Update centroid (running average)
        old_centroid = fam.get("centroid", [])
        n = len(members)
        if old_centroid and len(old_centroid) == len(bucket_emb):
            new_centroid = [
                (old_centroid[i] * (n - 1) + bucket_emb[i]) / n
                for i in range(len(bucket_emb))
            ]
            fam["centroid"] = new_centroid
        else:
            fam["centroid"] = bucket_emb

        fam["updated_at"] = datetime.now(timezone.utc).isoformat()
        fam["member_count"] = len(members)

        # Update emotion trend
        self._update_emotion_trend(fam, metadata)

        self._save_families()

        # Trigger summary update if enough members
        if len(members) >= self.min_summary_size:
            try:
                await self._update_summary(family_id)
            except Exception as e:
                logger.warning(f"Family summary update failed for {family_id}: {e}")

    async def _try_form_family_from_orphans(
        self, new_bucket_id: str, new_emb: list, new_meta: dict
    ) -> Optional[str]:
        """
        Check if this new orphan bucket is similar enough to other orphans
        to form a new family.
        """
        if not self.embedding_engine or not self.embedding_engine.enabled:
            return None

        # Find all orphan buckets (not in any family)
        all_family_members = set()
        for fam in self.families.values():
            all_family_members.update(fam.get("members", []))

        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception:
            return None

        orphan_buckets = [
            b for b in all_buckets
            if b["id"] not in all_family_members
            and b["id"] != new_bucket_id
            and b["metadata"].get("type") not in ("feel", "plan", "letter", "permanent")
            and not b["metadata"].get("pinned")
            and not b["metadata"].get("resolved")
        ]

        # Find orphans similar to the new bucket
        similar_orphans = []
        for ob in orphan_buckets:
            ob_emb = await self.embedding_engine.get_embedding(ob["id"])
            if ob_emb is None:
                continue
            sim = self._cosine_similarity(new_emb, ob_emb)
            if sim >= ORPHAN_CLUSTER_THRESHOLD:
                similar_orphans.append((ob, ob_emb, sim))

        if len(similar_orphans) < MIN_ORPHANS_FOR_NEW_FAMILY:
            return None

        # Sort by similarity, take top matches
        similar_orphans.sort(key=lambda x: x[2], reverse=True)
        selected = similar_orphans[:5]  # Max 5 initial members

        # Create new family
        family_id = f"fam_{int(time.time() * 1000) % 10000000:07d}"
        members = [new_bucket_id] + [ob["id"] for ob, _, _ in selected]

        # Compute centroid
        all_embs = [new_emb] + [emb for _, emb, _ in selected]
        centroid = self._compute_centroid(all_embs)

        # Compute initial emotion trend
        all_metas = [new_meta] + [ob["metadata"] for ob, _, _ in selected]
        emotion_trend = self._compute_emotion_trend(all_metas)

        self.families[family_id] = {
            "id": family_id,
            "name": "",  # Will be set by summary generation
            "summary": "",
            "members": members,
            "centroid": centroid,
            "member_count": len(members),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "emotion_trend": emotion_trend,
        }
        self._save_families()

        # Generate summary if enough members
        if len(members) >= self.min_summary_size:
            try:
                await self._update_summary(family_id)
            except Exception as e:
                logger.warning(f"Initial family summary failed for {family_id}: {e}")

        return family_id

    # ============================================================
    # Summary generation
    # ============================================================

    async def _update_summary(self, family_id: str):
        """Generate or update the narrative summary for a family."""
        fam = self.families.get(family_id)
        if not fam:
            return

        members = fam.get("members", [])
        if len(members) < self.min_summary_size:
            return

        # Collect member contents (sorted by created time)
        member_data = []
        for mid in members:
            try:
                bucket = await self.bucket_mgr.get(mid)
                if bucket:
                    created = bucket["metadata"].get("created", "")
                    content = bucket["content"][:300]
                    v = bucket["metadata"].get("valence", 0.5)
                    a = bucket["metadata"].get("arousal", 0.3)
                    member_data.append({
                        "created": created,
                        "content": content,
                        "valence": v,
                        "arousal": a,
                    })
            except Exception:
                continue

        if not member_data:
            return

        # Sort by time
        member_data.sort(key=lambda x: x.get("created", ""))

        # Build prompt for summary generation
        old_summary = fam.get("summary", "")
        is_update = bool(old_summary)

        member_texts = []
        for i, md in enumerate(member_data):
            member_texts.append(
                f"[{md['created'][:10]}] (v={md['valence']}, a={md['arousal']}) {md['content']}"
            )
        members_block = "\n".join(member_texts)

        if is_update:
            prompt = f"""你是记忆摘要器。这个家族已有摘要，现在有新成员加入，请增量更新摘要。

旧摘要：
{old_summary}

全部成员记忆（按时间排序）：
{members_block}

要求：
1. 用叙事体而非列表，按时间线讲述这组记忆的故事
2. 包含情感变化趋势（从什么情绪到什么情绪）
3. 控制在200-300字
4. 给这个家族起一个5字以内的名字

输出纯JSON：
{{"name": "家族名", "summary": "叙事摘要"}}"""
        else:
            prompt = f"""你是记忆摘要器。请为以下一组相关记忆生成叙事摘要。

成员记忆（按时间排序）：
{members_block}

要求：
1. 用叙事体而非列表，按时间线讲述这组记忆的故事
2. 包含情感变化趋势（从什么情绪到什么情绪）
3. 控制在200-300字
4. 给这个家族起一个5字以内的名字

输出纯JSON：
{{"name": "家族名", "summary": "叙事摘要"}}"""

        try:
            result = await self.dehydrator.llm_call(prompt)
            if result:
                parsed = self._parse_summary_result(result)
                if parsed:
                    fam["name"] = parsed.get("name", fam.get("name", ""))
                    fam["summary"] = parsed.get("summary", "")
                    fam["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._save_families()
                    logger.info(f"Family {family_id} summary updated: {fam['name']}")
        except Exception as e:
            logger.warning(f"Summary generation LLM call failed: {e}")

    def _parse_summary_result(self, raw: str) -> Optional[dict]:
        """Parse JSON from LLM summary response."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return {
                    "name": str(result.get("name", ""))[:10],
                    "summary": str(result.get("summary", ""))[:MAX_SUMMARY_LENGTH],
                }
        except (json.JSONDecodeError, Exception):
            logger.warning(f"Summary JSON parse failed: {raw[:200]}")
        return None

    # ============================================================
    # Fact extraction (called during hold)
    # ============================================================

    async def extract_facts(self, content: str) -> list[dict]:
        """
        Extract atomic facts from a memory content.
        Returns list of {subject, action, object, time, emotion}.
        """
        prompt = f"""你是事实抽取器。请把以下记忆拆分为原子事实单元。

记忆内容：
{content[:1000]}

每条事实格式：
{{"subject": "主体", "action": "动作/关系", "object": "对象", "time": "时间(如有)", "emotion": "情绪标签"}}

要求：
1. 一条记忆可拆为多条事实
2. 动作用动词维度标签：做了/说过/想要/喜欢/害怕/经历了/去了/买了/承诺/拥有/讨厌/发现/有/告诉/学过/拒绝/画了/梦到/做过/需要 等
3. 每条事实要尽量原子化，一个事实只描述一件事
4. 如果没有明确时间，time留空字符串

输出纯JSON数组：
[{{"subject": "...", "action": "...", "object": "...", "time": "...", "emotion": "..."}}]"""

        try:
            result = await self.dehydrator.llm_call(prompt)
            if result:
                return self._parse_facts_result(result)
        except Exception as e:
            logger.warning(f"Fact extraction failed: {e}")
        return []

    def _parse_facts_result(self, raw: str) -> list[dict]:
        """Parse fact extraction JSON."""
        try:
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
            if isinstance(result, list):
                facts = []
                for item in result[:10]:  # Max 10 facts per memory
                    if isinstance(item, dict):
                        facts.append({
                            "subject": str(item.get("subject", ""))[:50],
                            "action": str(item.get("action", ""))[:20],
                            "object": str(item.get("object", ""))[:100],
                            "time": str(item.get("time", ""))[:30],
                            "emotion": str(item.get("emotion", ""))[:20],
                        })
                return facts
        except (json.JSONDecodeError, Exception):
            logger.warning(f"Facts JSON parse failed: {raw[:200]}")
        return []

    # ============================================================
    # Emotion trend
    # ============================================================

    def _update_emotion_trend(self, fam: dict, new_meta: dict):
        """Update the family's emotion trend with a new member."""
        trend = fam.get("emotion_trend", [])
        v = float(new_meta.get("valence", 0.5))
        a = float(new_meta.get("arousal", 0.3))
        created = new_meta.get("created", datetime.now(timezone.utc).isoformat())
        trend.append({"time": created, "valence": v, "arousal": a})
        # Keep last 20 data points
        fam["emotion_trend"] = trend[-20:]

    def _compute_emotion_trend(self, metas: list[dict]) -> list[dict]:
        """Compute initial emotion trend from member metadata list."""
        trend = []
        for meta in metas:
            v = float(meta.get("valence", 0.5))
            a = float(meta.get("arousal", 0.3))
            created = meta.get("created", "")
            trend.append({"time": created, "valence": v, "arousal": a})
        trend.sort(key=lambda x: x.get("time", ""))
        return trend[-20:]

    # ============================================================
    # Retrieval: surface families for breath
    # ============================================================

    async def surface_families(self, max_results: int = 3, max_tokens: int = 3000) -> list[str]:
        """
        Return formatted family summaries for breath surfacing.
        Only families with summaries are surfaced.
        """
        surfaceable = [
            (fid, fam) for fid, fam in self.families.items()
            if fam.get("summary")
        ]
        if not surfaceable:
            return []

        # Sort by recency (updated_at)
        surfaceable.sort(key=lambda x: x[1].get("updated_at", ""), reverse=True)

        results = []
        for fid, fam in surfaceable[:max_results]:
            name = fam.get("name", "未命名")
            summary = fam.get("summary", "")
            member_count = fam.get("member_count", 0)
            trend = fam.get("emotion_trend", [])
            trend_str = ""
            if len(trend) >= 2:
                first = trend[0]
                last = trend[-1]
                trend_str = f" | 情感趋势: v({first['valence']:.1f}→{last['valence']:.1f}) a({first['arousal']:.1f}→{last['arousal']:.1f})"
            results.append(
                f"🏠 [家族:{name}] [family_id:{fid}] ({member_count}条){trend_str}\n{summary}"
            )

        return results

    async def search_families(self, query: str, max_results: int = 3) -> list[str]:
        """
        Search families by query similarity.
        Returns formatted family summaries.
        """
        if not self.embedding_engine or not self.embedding_engine.enabled:
            return []

        try:
            query_emb = await self.embedding_engine._generate_embedding(query)
            if not query_emb:
                return []
        except Exception:
            return []

        scored = []
        for fid, fam in self.families.items():
            centroid = fam.get("centroid")
            if not centroid:
                continue
            sim = self._cosine_similarity(query_emb, centroid)
            if sim >= 0.5:  # minimum relevance
                scored.append((fid, fam, sim))

        scored.sort(key=lambda x: x[2], reverse=True)

        results = []
        for fid, fam, sim in scored[:max_results]:
            name = fam.get("name", "未命名")
            summary = fam.get("summary", "")
            member_count = fam.get("member_count", 0)
            if summary:
                results.append(
                    f"🏠 [家族:{name}] [family_id:{fid}] (相似度:{sim:.2f}, {member_count}条)\n{summary}"
                )
            else:
                # No summary yet, list member IDs
                members = fam.get("members", [])[:5]
                results.append(
                    f"🏠 [家族:{name}] [family_id:{fid}] (相似度:{sim:.2f}, {member_count}条)\n成员: {', '.join(members)}"
                )

        return results

    async def expand_family(self, family_id: str) -> str:
        """
        Expand a family to show all member details.
        Called when user/model wants to see family contents.
        """
        fam = self.families.get(family_id)
        if not fam:
            return f"家族 {family_id} 不存在。"

        name = fam.get("name", "未命名")
        summary = fam.get("summary", "")
        members = fam.get("members", [])
        trend = fam.get("emotion_trend", [])

        parts = [f"=== 家族：{name} ({len(members)}条) ==="]
        if summary:
            parts.append(f"摘要：{summary}")
        if len(trend) >= 2:
            first = trend[0]
            last = trend[-1]
            parts.append(
                f"情感趋势：v({first['valence']:.1f}→{last['valence']:.1f}) "
                f"a({first['arousal']:.1f}→{last['arousal']:.1f})"
            )

        parts.append("\n--- 成员记忆 ---")
        for mid in members:
            try:
                bucket = await self.bucket_mgr.get(mid)
                if bucket:
                    content_preview = bucket["content"][:200].strip()
                    v = bucket["metadata"].get("valence", 0.5)
                    a = bucket["metadata"].get("arousal", 0.3)
                    created = bucket["metadata"].get("created", "")[:10]
                    parts.append(f"[{created}] [bucket_id:{mid}] (v={v}, a={a}) {content_preview}")
            except Exception:
                parts.append(f"[bucket_id:{mid}] (无法读取)")

        return "\n".join(parts)

    # ============================================================
    # Lines: cross-cutting dimensions
    # ============================================================

    async def get_lines(self) -> dict:
        """
        Get all cross-cutting lines (verb/topic dimensions) across families.
        Returns {action_verb: [{family_id, bucket_id, object, score}]}
        """
        lines = {}
        for fid, fam in self.families.items():
            facts = fam.get("facts", [])
            for fact in facts:
                action = fact.get("action", "")
                if not action:
                    continue
                if action not in lines:
                    lines[action] = []
                lines[action].append({
                    "family_id": fid,
                    "subject": fact.get("subject", ""),
                    "object": fact.get("object", ""),
                    "emotion": fact.get("emotion", ""),
                })
        return lines

    async def store_facts_for_family(self, family_id: str, facts: list[dict]):
        """Store extracted facts in the family data."""
        fam = self.families.get(family_id)
        if not fam:
            return
        existing_facts = fam.get("facts", [])
        existing_facts.extend(facts)
        # Deduplicate by action+object
        seen = set()
        unique_facts = []
        for f in existing_facts:
            key = f"{f.get('action','')}-{f.get('object','')}"
            if key not in seen:
                seen.add(key)
                unique_facts.append(f)
        fam["facts"] = unique_facts[-50:]  # Keep last 50
        self._save_families()

    # ============================================================
    # Structural type detection (P3)
    # ============================================================

    def detect_structure_type(self, analysis: dict, content: str) -> str:
        """
        Detect the structural type of a memory from content analysis.
        Returns: 'fact' | 'promise' | 'desire' | 'fear' | 'tension' | 'decision'
        """
        content_lower = content.lower()
        tags = [t.lower() for t in (analysis.get("tags") or [])]
        arousal = float(analysis.get("arousal", 0.3))

        # Promise/plan detection
        promise_keywords = ["打算", "计划", "准备", "要去", "承诺", "约定", "会", "一定", "保证"]
        if any(kw in content for kw in promise_keywords):
            return "promise"

        # Desire detection
        desire_keywords = ["想要", "想", "渴望", "希望", "梦想", "期待", "盼", "想念"]
        if any(kw in content for kw in desire_keywords):
            return "desire"

        # Fear detection
        fear_keywords = ["害怕", "担心", "恐惧", "焦虑", "怕", "紧张", "不安"]
        if any(kw in content for kw in fear_keywords):
            return "fear"

        # Tension/conflict detection (high arousal + mixed emotion)
        if arousal >= 0.7:
            return "tension"

        # Decision detection
        decision_keywords = ["决定", "选择", "放弃", "不再", "开始"]
        if any(kw in content for kw in decision_keywords):
            return "decision"

        return "fact"

    # ============================================================
    # Spaced revisit (P1)
    # ============================================================

    def get_revisit_schedule(self, importance: int) -> list[int]:
        """
        Return revisit schedule in days based on importance.
        Higher importance = more revisit points.
        """
        if importance >= 9:
            return [1, 3, 7, 14]
        elif importance >= 7:
            return [1, 3, 7]
        elif importance >= 5:
            return [3, 7]
        else:
            return []

    async def get_due_revisits(self) -> list[dict]:
        """
        Find buckets that are due for revisit based on their schedule.
        Called by dream().
        """
        due = []
        try:
            all_buckets = await self.bucket_mgr.list_all(include_archive=False)
        except Exception:
            return due

        now = datetime.now(timezone.utc)

        for b in all_buckets:
            meta = b["metadata"]
            if meta.get("resolved") or meta.get("type") in ("feel", "plan", "letter"):
                continue

            schedule = meta.get("revisit_schedule", [])
            if not schedule:
                continue

            created_str = meta.get("created", "")
            if not created_str:
                continue

            try:
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            except Exception:
                continue

            last_revisit = meta.get("last_revisit", "")
            revisited_days = set(meta.get("revisited_days", []))

            for day in schedule:
                if day in revisited_days:
                    continue
                target = created.replace(tzinfo=timezone.utc) if created.tzinfo is None else created
                days_since = (now - target).days
                if days_since >= day:
                    due.append({
                        "bucket": b,
                        "due_day": day,
                        "days_since_created": days_since,
                    })
                    break  # Only the earliest unvisited day

        return due

    # ============================================================
    # API endpoints data
    # ============================================================

    def get_all_families(self) -> list[dict]:
        """Return all families for API/frontend."""
        result = []
        for fid, fam in self.families.items():
            result.append({
                "id": fid,
                "name": fam.get("name", ""),
                "summary": fam.get("summary", ""),
                "member_count": fam.get("member_count", 0),
                "members": fam.get("members", []),
                "emotion_trend": fam.get("emotion_trend", []),
                "facts": fam.get("facts", []),
                "created_at": fam.get("created_at", ""),
                "updated_at": fam.get("updated_at", ""),
            })
        result.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return result

    def get_family(self, family_id: str) -> Optional[dict]:
        """Return a single family's data."""
        fam = self.families.get(family_id)
        if not fam:
            return None
        return {
            "id": family_id,
            "name": fam.get("name", ""),
            "summary": fam.get("summary", ""),
            "member_count": fam.get("member_count", 0),
            "members": fam.get("members", []),
            "emotion_trend": fam.get("emotion_trend", []),
            "facts": fam.get("facts", []),
            "created_at": fam.get("created_at", ""),
            "updated_at": fam.get("updated_at", ""),
        }

    def remove_bucket_from_families(self, bucket_id: str):
        """Remove a bucket from all families (called on bucket delete)."""
        changed = False
        for fam in self.families.values():
            members = fam.get("members", [])
            if bucket_id in members:
                members.remove(bucket_id)
                fam["members"] = members
                fam["member_count"] = len(members)
                changed = True
        if changed:
            self._save_families()

    # ============================================================
    # Utilities
    # ============================================================

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
        if not embeddings:
            return []
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i in range(dim):
                centroid[i] += emb[i]
        n = len(embeddings)
        return [c / n for c in centroid]
