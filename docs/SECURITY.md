# NexoraLM Security & Data Integrity Policy

## Secrets Management
- Kaggle API tokens and any remote credentials must never be committed to Git.
- Credentials are stored in `~/.kaggle/access_token` and referenced via local environment variables.
- Git excludes `.env` and `*.pt` files via `.gitignore`.

## Data Pipeline Integrity
- Data partitions (train, validation, test) are segregated with deterministic MD5 hashing prior to tokenization to guarantee zero leakage.
- Pretraining corpus filtering strips empty, repetitive, and malformed inputs to prevent corrupted gradient updates.
- API serving implements input validation via Pydantic models to prevent prompt injection or malformed payload crashes.
