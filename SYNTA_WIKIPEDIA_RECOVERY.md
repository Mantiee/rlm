# Wikipedia importer recovery

Observed: v82 legacy importer got HTTP 403 from its first dumpstatus request, before any corpus download. Wikimedia's June 2026 policy requires an identifiable User-Agent on automated dump downloads. Generic Python urllib headers are a likely cause, not proof that this host has no other access restriction.

Changed files:
- `rlm/v100/wikimedia_exports.py`: explicit SyntaKnowledgeIndexer identity with project contact URL; official mediawiki_content_current monthly SHA256SUMS manifests; current month then up to five older months only on manifest 404; strict part filenames and hashes; SHA-256 before indexing; persistent progress/failure receipt under `research/knowledge/import-status.json`. A 403/429 stops without identity/address rotation or evasive retries.
- `rlm/v100/knowledge_rag.py`: indexing progress every 1000 articles. Same disk index/schema and source/revision attribution, no weights changed.
- `tools/run-synta-wikipedia.sh`: download the two pinned standalone stdlib modules into `ai-v100/tools/wiki-import/REV`, compile-check them, run the importer. No package upgrade, mission restart, GPU reservation or training interruption.
- `tests/test_v100_wikimedia_exports.py`: request identity, no forbidden retry, completed prior-month selection, path confinement and checksum admission.

Publisher guidance:
https://diff.wikimedia.org/2026/06/29/tech-news-2026-issue-27/
https://wikitech.wikimedia.org/wiki/MediaWiki_Content_File_Exports

Validation: 7 targeted importer/RAG tests passed; Ruff, Bash syntax and diff checks passed. Live provider access and full corpus ingestion are not verified in repository tests. Cached completed parts are hash-checked and reused; partial download restarts that part, and reindexing deterministically replaces source passages without duplication. Do not claim a full corpus or weight learning until the complete receipt exists.

Use the standalone launcher from this exact revision. The installed v82 `python -m rlm.v100.wiki_ingest` remains the legacy module and must not be used for this recovery. The imported index is directly available to the existing master `search_knowledge` tool.
