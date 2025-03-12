import dotenv
from valyu import Valyu

# Load environment variables (in case API keys are stored there)
dotenv.load_dotenv()

def test_valyu_api():
    """Simple test function for the Valyu API."""
    print("Initializing Valyu client...")
    valyu = Valyu()
    
    # Test query
    test_query = "What are the key components of a blockchain system?"
    print(f"Sending test query: '{test_query}'")
    
    try:
        # Call the context method
        response = valyu.context(
            query=test_query,
            search_type="all",
            max_num_results=3,
            max_price=50,
            similarity_threshold=0.4
        )
        
        # Print response summary
        print(f"\nReceived {len(response.results)} results:")
        print(response)
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_valyu_api()
