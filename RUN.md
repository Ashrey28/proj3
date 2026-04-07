# Stories 3 and 9 add-on

I did not edit the original source lines. I added new files only.

## Run

From the `proj3-main/proj3-main` folder:

```bash
python -m uvicorn app:app --reload
```

## Pages

- `/` chat UI for Story 3
- `/developer` developer dashboard for Story 9

## What was added

- `app.py` new FastAPI entrypoint
- `rag.py` dataset loader + retriever
- `grounding.py` grounded answer engine
- `evaluation.py` evaluation metrics
- `static/index.html` end-user UI
- `static/developer.html` developer/admin UI
- `data/knowledge_base.json` starter dataset
- `data/evaluation_set.json` starter evaluation set

## Notes

- This works with `USE_LOCAL_CLASSIFIER=true` for local testing.
- If you provide an OpenAI API key and do not enable local mode, grounded answer generation will use the model while still restricting it to retrieved context.
