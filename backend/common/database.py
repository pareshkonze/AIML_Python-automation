import os
import asyncpg
import logging
from .models import Location

logger = logging.getLogger(__name__)

db_config = {
    'host': os.getenv('DB_HOST'),
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': os.getenv('DB_PORT')
}

pool = None

async def init_db_pool():
    global pool
    if pool is None:
        try:
            pool = await asyncpg.create_pool(**db_config, min_size=1, max_size=10)
            logger.info("Database pool initialized")
        except Exception as e:
            logger.error(f"Error initializing database pool: {e}", exc_info=True)
            raise

async def close_db_pool():
    global pool
    if pool:
        await pool.close()
        logger.info("Database pool closed")

async def execute_query(query, *args):
    global pool
    if pool is None:
        await init_db_pool()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)

async def execute_procedure(procedure_name, *args):
    global pool
    if pool is None:
        await init_db_pool()
    async with pool.acquire() as conn:
        try:
            await conn.execute(f"CALL {procedure_name}($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18)", *args)
        except asyncpg.exceptions.UniqueViolationError:
            logger.warning(f"Duplicate entry detected in {procedure_name}. Skipping.")
        except Exception as e:
            logger.error(f"Error executing procedure {procedure_name}: {e}", exc_info=True)
            raise

async def get_locations():
    query = 'SELECT id, location_id, location_name, website_name FROM tbl_location'
    results = await execute_query(query)
    return [Location(**dict(row)) for row in results]