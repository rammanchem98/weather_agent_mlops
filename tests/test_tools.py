from unittest.mock import patch, MagicMock
from src.tools.vector_search import search_air_quality_db

@patch("src.tools.vector_search.qdrant_client")
@patch("src.tools.vector_search.client")
def test_search_air_quality_db_returns_joined_text(mock_genai_client, mock_qdrant_client):
    # mock the embedding call
    mock_embedding_response = MagicMock()
    mock_embedding_response.embeddings = [MagicMock(values=[0.1] * 768)]
    mock_genai_client.models.embed_content.return_value = mock_embedding_response

    # mock the Qdrant search result — two fake points with payload text
    point_1 = MagicMock()
    point_1.payload = {"text": "city:London,temp:24C,pm25:2.65"}
    point_2 = MagicMock()
    point_2.payload = {"text": "city:London,temp:18C,pm25:1.1"}
    mock_qdrant_client.query_points.return_value = MagicMock(points=[point_1, point_2])

    result = search_air_quality_db("historical air quality in London")

    assert "city:London,temp:24C,pm25:2.65" in result
    assert "city:London,temp:18C,pm25:1.1" in result
    mock_qdrant_client.query_points.assert_called_once()

@patch("src.tools.vector_search.qdrant_client")
@patch("src.tools.vector_search.client")
def test_search_air_quality_db_handles_no_results(mock_genai_client, mock_qdrant_client):
    mock_embedding_response = MagicMock()
    mock_embedding_response.embeddings = [MagicMock(values=[0.1] * 768)]
    mock_genai_client.models.embed_content.return_value = mock_embedding_response

    mock_qdrant_client.query_points.return_value = MagicMock(points=[])  # empty — no data found

    result = search_air_quality_db("historical air quality in Atlantis")

    assert result == ""  # confirms the "empty result, not a crash" behavior we relied on in the docstring fix