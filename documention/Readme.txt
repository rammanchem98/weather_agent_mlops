To pull and import the predefined qdrant docker image

docker run -d -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage" qdrant/qdrant

To verify the image 
http://localhost:6333/dashboard

change the path in config dev
add the qdrant path in vector search
run the index qdrant again to pull the raw data into qdrant collection by embedding with the help of gemini model

To remove local qdrant
Remove-Item -Recurse -Force src\data\local_qdrant_storage

test the everything Is working fine
python -m pytest tests/ -v
python -m eval.run_eval
python main.py