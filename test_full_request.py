#!/usr/bin/env python3
"""
Test the full Postman request with all 10 questions
"""

import requests
import json

def test_full_request():
    """Test the complete API request"""
    print("🧪 Testing full API request with all 10 questions...")
    
    url = "http://localhost:8000/hackrx/run"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 031e2883dcfac08106d5a9982528deff7dcd207bd1efbca391476ea56fec65ac"
    }
    
    data = {
        "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
        "questions": [
            "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
            "What is the waiting period for pre-existing diseases (PED) to be covered?",
            "Does this policy cover maternity expenses, and what are the conditions?",
            "What is the waiting period for cataract surgery?",
            "Are the medical expenses for an organ donor covered under this policy?",
            "What is the No Claim Discount (NCD) offered in this policy?",
            "Is there a benefit for preventive health check-ups?",
            "How does the policy define a 'Hospital'?",
            "What is the extent of coverage for AYUSH treatments?",
            "Are there any sub-limits on room rent and ICU charges for Plan A?"
        ]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Full API test successful!")
            print("\n📋 Answers:")
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"{i}. {answer}")
                print("-" * 80)
        else:
            print(f"❌ API test failed: {response.status_code}")
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_full_request() 