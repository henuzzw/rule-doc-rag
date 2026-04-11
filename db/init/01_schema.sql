CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS requirements (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  business_background TEXT NOT NULL DEFAULT '',
  source_text TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
  id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  source_id TEXT NOT NULL,
  title TEXT NOT NULL,
  content TEXT NOT NULL,
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding VECTOR(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_source
  ON knowledge_chunks (source_type, source_id);

CREATE INDEX IF NOT EXISTS idx_knowledge_chunks_embedding_hnsw
  ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS bad_cases (
  id TEXT PRIMARY KEY,
  rule_code TEXT NOT NULL,
  rule_name TEXT NOT NULL,
  title TEXT NOT NULL,
  bad_summary TEXT NOT NULL,
  failure_reason TEXT NOT NULL,
  corrected_hint TEXT NOT NULL DEFAULT '',
  metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding VECTOR(1024) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bad_cases_rule_code
  ON bad_cases (rule_code, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_bad_cases_embedding_hnsw
  ON bad_cases USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS generated_documents (
  id TEXT PRIMARY KEY,
  rule_code TEXT NOT NULL,
  rule_name TEXT NOT NULL,
  requirement_id TEXT NOT NULL REFERENCES requirements(id),
  status TEXT NOT NULL DEFAULT 'draft',
  doc_json JSONB NOT NULL,
  prompt TEXT NOT NULL,
  raw_response TEXT NOT NULL,
  retrieved_context JSONB NOT NULL DEFAULT '[]'::jsonb,
  reviewer_notes TEXT NOT NULL DEFAULT '',
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_generated_documents_requirement
  ON generated_documents (requirement_id, created_at DESC);

