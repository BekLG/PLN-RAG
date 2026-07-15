import asyncio
from config import Settings
from core.service import PLNRAGService
from parsers.langextract_pln_parser import LangExtractPLNParser

async def main():
    parser = LangExtractPLNParser()
    service = PLNRAGService(parser)
    result = await service.debug_query("Is Abebe consuming excessive carbohydrates?")
    print(result.model_dump_json(indent=2))

asyncio.run(main())
