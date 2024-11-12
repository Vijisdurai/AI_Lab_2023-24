# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 28/10/2024                                                                           
### REGISTER NUMBER : 212222220057 (Vijis Durai R)
### AIM: 
To write a program to train the classifier for a real-time product recommendation system using Named Entity Recognition (NER) with Hugging Face transformers.
###  Algorithm:
1. Data Loading: Prepare the text data (user queries) for classification.
2. NER Model Setup: Load a pre-trained Named Entity Recognition (NER) model using Hugging Face transformers.
3. Feature Extraction: Use rule-based methods to extract relevant features from user queries, including product type, budget, and specific features.
4. Prediction: Use the trained model to predict and classify user requirements based on extracted entities.
5. Recommendation Generation: Use classified results to suggest relevant product recommendations.
### Program:
```
pip install tranformers , requests
```
```
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# Load a pretrained Hugging Face tokenizer and model for Named Entity Recognition (NER)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

def extract_user_input_huggingface(user_query):
    """
    This function processes the user query using rule-based extraction and Hugging Face NER.
    Extracts product type, budget, and features.
    """
    product_type = None
    budget = None
    features = []

    # Make user query lowercased for case-insensitive matching
    user_query = user_query.lower()

    # Direct keyword matching for product types
    if "smartphone" in user_query:
        product_type = "smartphone"
    elif "laptop" in user_query or "notebook" in user_query:
        product_type = "laptop"
    elif "tablet" in user_query:
        product_type = "tablet"
    elif "tv" in user_query or "television" in user_query:
        product_type = "tv"

    # Detect budget using rule-based approach
    words = user_query.split()
    for i, word in enumerate(words):
        if word == "under" and i + 1 < len(words):
            # Handle cases like '10k' (convert '10k' to '10000')
            budget_str = words[i + 1].replace('$', '').replace(',', '').replace('.', '').strip().lower()
            if 'k' in budget_str:
                budget = int(float(budget_str.replace('k', '')) * 1000)  # Convert '10k' to '10000'
            else:
                budget = int(budget_str)

    # Detect features (expandable to other features)
    if "camera" in user_query:
        features.append("good camera")
    if "battery" in user_query:
        features.append("long battery life")
    if "screen" in user_query:
        features.append("large screen")
    if "performance" in user_query:
        features.append("performance")

    return product_type, budget, features
# Main script to take user input
if __name__ == "__main__":
    # Dynamically prompt user for query
    user_query = input("Please enter your product requirements (e.g., 'smartphone under $500 with good camera'): ")

    # Process the user query using Hugging Face and rule-based extraction
    product_type, budget, features = extract_user_input_huggingface(user_query)

    # Display the extracted information
    print(f"\nExtracted Information:")
    print(f"Product Type: {product_type}")
    print(f"Budget: ${budget}")
    print(f"Features: {features}")
```
```
import requests

def product_details(asin, rapidapi_key):
    """
    Fetch product details using the Amazon API from RapidAPI.
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product/{asin}"

    headers = {
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
        "x-rapidapi-key": rapidapi_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching product details. Status code: {response.status_code}")
        return None
```

```
# api_services/product_offers.py
import requests

def product_offers(asin, rapidapi_key):
    """
    Fetch offers for a product using the Amazon API on RapidAPI.
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product/{asin}/offers"

    headers = {
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
        "x-rapidapi-key": rapidapi_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching product offers. Status code: {response.status_code}")
        return None
```

```
# api_services/product_reviews.py
import requests

def product_reviews(asin, rapidapi_key):
    """
    Fetch product reviews using the Amazon API on RapidAPI.
    """
    url = f"https://real-time-amazon-data.p.rapidapi.com/product/{asin}/reviews"

    headers = {
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
        "x-rapidapi-key": rapidapi_key
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching product reviews. Status code: {response.status_code}")
        return None

import requests

def product_search(query, rapidapi_key):
    """
    Perform product search using the RapidAPI endpoint for real-time product data.
    :param query: The product search query (e.g., "Phone", "Laptop")
    :param rapidapi_key: Your RapidAPI key
    :return: JSON response with product results
    """
    # Replace with the correct API endpoint
    url = "https://real-time-amazon-data.p.rapidapi.com/search"

    headers = {
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",  # Replace with the actual host
        "x-rapidapi-key": rapidapi_key  # Replace with your RapidAPI key
    }

    params = {
        "query": query,                 # The product type from user input (e.g., Phone)
        "page": "1",                    # Page number
        "country": "US",                # Country for search
        "sort_by": "RELEVANCE",         # Sorting preference
        "product_condition": "ALL",     # All product conditions
        "is_prime": "false"             # Whether to filter for Prime products
    }

    # Perform the GET request to the API
    response = requests.get(url, headers=headers, params=params)

    # Check if the response is successful
    if response.status_code == 200:
        return response.json()  # Return the response data as JSON
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
```

```
# utils/filters.py
def filter_by_price(products, max_price):
    """
    Filter products that are below or equal to the max_price.
    """
    return [product for product in products if float(product['product_price'].replace('$', '').replace(',', '')) <= max_price]

def filter_by_rating(products, min_rating):
    """
    Filter products that have a rating greater than or equal to min_rating.
    """
    return [product for product in products if float(product.get('product_star_rating', 0)) >= min_rating]

def filter_by_availability(products):
    """
    Filter products that are in stock.
    """
    return [product for product in products if "in stock" in product.get('product_availability', '').lower()]

def filter_by_offers(products):
    """
    Filter products that have offers or coupons.
    """
    return [product for product in products if product.get('coupon_text') or product.get('has_variations') == True]

def apply_filters(products, filters):
    """
    Apply a series of filters to the product list.
    filters: dictionary of filters (e.g., {'price': 500, 'rating': 4.0, 'availability': True, 'offers': True})
    """
    if 'price' in filters:
        products = filter_by_price(products, filters['price'])
    if 'rating' in filters:
        products = filter_by_rating(products, filters['rating'])
    if 'availability' in filters and filters['availability']:
        products = filter_by_availability(products)
    if 'offers' in filters and filters['offers']:
        products = filter_by_offers(products)

    return products
```

```
# Main script to take user input and search for products
if __name__ == "__main__":
    # Get the API key
    rapidapi_key = "4fdc28be8cmsh73bf9b8beceb081p1a4ec0jsn8702302e3376"

    # Dynamically prompt user for query
    user_query = input("Please enter your product requirements (e.g., 'smartphone under $500 with good camera'): ")

    # Process the user query using Hugging Face and rule-based extraction
    product_type, budget, features = extract_user_input_huggingface(user_query)

    # Display the extracted information
    print(f"\nExtracted Information:")
    print(f"Product Type: {product_type}")
    print(f"Budget: ${budget}")
    print(f"Features: {features}")

    # Perform product search using the extracted query
    # Perform product search using the extracted query
    # Perform product search using the extracted query
    # Perform product search using the extracted query
if product_type:
    search_query = f"{product_type} under ${budget} with {' and '.join(features)}"
    product_data = product_search(search_query, rapidapi_key)

    if product_data:
        # Use .get() to avoid KeyError and access the correct key 'data' -> 'products'
        products = product_data.get('data', {}).get('products', None)
        if products:
            print("\nProduct Search Results:")
            for product in products:
                title = product.get('product_title', 'N/A')
                price = product.get('product_price', 'N/A')
                rating = product.get('product_star_rating', 'N/A')
                num_ratings = product.get('product_num_ratings', 'N/A')
                availability = product.get('product_availability', 'N/A')
                delivery = product.get('delivery', 'N/A')

                print(f"Title: {title}")
                print(f"Price: {price}")
                print(f"Rating: {rating} ({num_ratings} ratings)")
                print(f"Availability: {availability}")
                print(f"Delivery: {delivery}")
                print("-" * 30)
        else:
            print("No products found in the response. Please check the query or API response.")
    else:
        print("No product data found.")
else:
    print("Unable to determine product type.")
```

### Output:
![image](https://github.com/user-attachments/assets/05dd13d6-e2c7-4bb9-a671-320c5b5e4542)

### Result:
Thus the system was trained successfully and the prediction was carried out.
