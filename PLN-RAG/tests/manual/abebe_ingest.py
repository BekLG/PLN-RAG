import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.orchestration.service import PLNRAGService
from parsers.langextract_pln_parser import LangExtractPLNParser

async def main():
    parser = LangExtractPLNParser()
    service = PLNRAGService(parser)
    service.reset("all")
    text = """People who frequently eat pasta tend to consume higher amounts of carbohydrates, especially if the portions are large and the pasta is refined rather than whole grain. However, high carbohydrate intake only leads to obesity when total calorie intake exceeds energy expenditure over a sustained period. Regular physical activity, balanced diet, and metabolic differences can reduce this effect. Obesity increases the risk of developing type 2 diabetes, but not all obese individuals develop diabetes; genetic predisposition, age, and lifestyle factors also play a role. A person is considered at elevated diabetes risk if they are obese and have at least one additional factor such as a sedentary lifestyle, family history of diabetes, or consistently high sugar intake.

Abebe eats pasta almost every day, often in large portions, and prefers refined pasta over whole grain. He works a desk job and exercises only occasionally. However, he does not consume many sugary foods, and there is no known family history of diabetes. Recently, his weight has increased, but he has not been clinically classified as obese."""
    
    res = await service.ingest_batch([text])
    print("ATOMS:")
    for atom in res[0].atoms:
        print(atom)

if __name__ == "__main__":
    asyncio.run(main())
