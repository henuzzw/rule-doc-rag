from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import db_session  # noqa: E402


def main() -> None:
    schema_sql = (PROJECT_ROOT / "db" / "init" / "01_schema.sql").read_text(
        encoding="utf-8"
    )
    with db_session() as conn:
        conn.execute(schema_sql)
    print("database schema initialized")


if __name__ == "__main__":
    main()

