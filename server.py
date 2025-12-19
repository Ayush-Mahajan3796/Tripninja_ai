from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import random
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import requests
from contextlib import asynccontextmanager
import json

# LangChain imports for agentic system
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 2. DEFINE THE LIFESPAN HANDLER
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    """
    logger.info("Application is starting up...")
    
    yield  # The application runs
    
    # Shutdown logic
    logger.info("Application is shutting down... Closing MongoDB client.")
    client.close()


# 3. PASS THE LIFESPAN HANDLER TO FASTAPI
app = FastAPI(lifespan=lifespan)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str

class TripRequest(BaseModel):
    pickup: str
    drop: str
    budget: float
    departDate: Optional[str] = None
    nights: int = 5
    vehicleType: str = "Car"
    roomType: str = "double"
    bookAccommodation: bool = False

# Tools/Helper Functions - Converted to LangChain tools for agentic system
@tool
def search_rides(pickup: str, drop: str, budget: float = None, vehicle_type: str = None) -> List[Dict[str, Any]]:
    """Search rides by calculating distance and estimating prices"""
    try:
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_headers = {"User-Agent": "TripNinjaAgent/1.0"}

        pickup_resp = requests.get(
            geo_url, params={"q": pickup, "format": "json"}, headers=geo_headers, timeout=10
        ).json()
        drop_resp = requests.get(
            geo_url, params={"q": drop, "format": "json"}, headers=geo_headers, timeout=10
        ).json()

        if not pickup_resp or not drop_resp:
            return []

        start = f"{pickup_resp[0]['lon']},{pickup_resp[0]['lat']}"
        end = f"{drop_resp[0]['lon']},{drop_resp[0]['lat']}"

        ors_url = "https://api.openrouteservice.org/v2/directions/driving-car"
        ors_headers = {"Authorization": os.getenv("ORS_API_KEY")}
        params = {"start": start, "end": end}

        response = requests.get(ors_url, headers=ors_headers, params=params, timeout=15)
        if response.status_code != 200:
            return []

        route_data = response.json()
        distance_km = route_data["features"][0]["properties"]["segments"][0]["distance"] / 1000
        duration_min = route_data["features"][0]["properties"]["segments"][0]["duration"] / 60

        rides = [
            {"provider": "Rapido", "type": "Bike", "price": max(distance_km * 10, 50), "eta": f"{duration_min:.0f} min", "distance": f"{distance_km:.0f} km"},
            {"provider": "Ola", "type": "Auto", "price": max(distance_km * 12, 60), "eta": f"{duration_min:.0f} min", "distance": f"{distance_km:.0f} km"},
            {"provider": "Uber", "type": "Car", "price": max(distance_km * 15, 100), "eta": f"{duration_min:.0f} min", "distance": f"{distance_km:.0f} km"}
        ]

        if vehicle_type:
            rides = [r for r in rides if r["type"] == vehicle_type]

        rides = [r for r in rides if budget is None or r["price"] <= budget]
        return rides

    except Exception as e:
        logging.error(f"Error in search_rides: {e}")
        return []

@tool
def get_weather(location: str) -> Dict[str, Any]:
    """Get current weather for a location"""
    try:
        api_key = os.getenv("OPENWEATHER_API_KEY")
        if not api_key:
            return {"description": "N/A", "temp": 0, "feelsLike": 0, "humidity": 0, "windSpeed": 0}

        url = "http://api.openweathermap.org/data/2.5/weather"
        params = {"q": location, "appid": api_key, "units": "metric"}

        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return {"description": "N/A", "temp": 0, "feelsLike": 0, "humidity": 0, "windSpeed": 0}

        data = resp.json()
        return {
            "description": data["weather"][0]["description"].capitalize(),
            "temp": round(data["main"]["temp"]),
            "feelsLike": round(data["main"]["feels_like"]),
            "humidity": data["main"]["humidity"],
            "windSpeed": round(data["wind"]["speed"] * 3.6)  # m/s to km/h
        }

    except Exception as e:
        logging.error(f"Error in get_weather: {e}")
        return {"description": "N/A", "temp": 0, "feelsLike": 0, "humidity": 0, "windSpeed": 0}

@tool
def generate_itinerary(pickup: str, drop: str) -> List[Dict[str, Any]]:
    """Generate tourist itinerary using OpenTripMap API - returns REAL tourist places at destination"""
    try:
        otm_key = os.getenv("OPENTRIPMAP_API_KEY")
        
        if not otm_key:
            logging.error("OPENTRIPMAP_API_KEY not found in environment variables")
            return []
        
        # Get destination coordinates
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "TripNinjaAgent/1.0"}
        
        drop_resp = requests.get(geo_url, params={"q": drop, "format": "json", "limit": 1}, headers=headers, timeout=10)
        
        if drop_resp.status_code != 200:
            logging.error(f"Geocoding failed for destination: {drop}")
            return []
        
        drop_data = drop_resp.json()
        if not drop_data:
            logging.error(f"Could not geocode destination: {drop}")
            return []

        # Use destination coordinates directly (not route midpoint)
        dest_lat = float(drop_data[0]['lat'])
        dest_lon = float(drop_data[0]['lon'])
        
        logging.info(f"Searching for tourist places near {drop} at coordinates: {dest_lat}, {dest_lon}")

        # Step 1: Get list of place IDs using radius search
        otm_radius_url = "https://api.opentripmap.com/0.1/en/places/radius"
        radius_params = {
            "radius": 10000,  # 10km radius around destination for better coverage
            "lon": dest_lon,
            "lat": dest_lat,
            "apikey": otm_key,
            "limit": 20,  # Get more places to filter
            "kinds": "interesting_places,cultural,historic,architecture,religion,natural,monuments"
        }
        
        radius_response = requests.get(otm_radius_url, params=radius_params, timeout=15)
        
        radius_data = {}
        if radius_response.status_code != 200:
            logging.error(f"OpenTripMap radius API failed with status {radius_response.status_code}: {radius_response.text}")
            # Set empty data to trigger Overpass fallback
            radius_data = {}
        else:
            logging.info(f"OpenTripMap API returned status 200")
            try:
                radius_data = radius_response.json()
            except:
                logging.warning("Failed to parse OpenTripMap JSON response")
                radius_data = {}
        logging.info(f"OpenTripMap radius response type: {type(radius_data)}, keys: {radius_data.keys() if isinstance(radius_data, dict) else 'N/A'}")
        logging.info(f"OpenTripMap response preview: {str(radius_data)[:500]}")
        
        # Check if response indicates an error
        if isinstance(radius_data, dict):
            if "error" in radius_data:
                logging.error(f"OpenTripMap API error: {radius_data.get('error')}")
            if "message" in radius_data:
                logging.warning(f"OpenTripMap API message: {radius_data.get('message')}")
        
        # Extract places from response - try multiple formats
        places_list = []
        place_ids = []
        
        if isinstance(radius_data, dict):
            if "found" in radius_data:
                # Response format: {"found": [{"xid": "...", "name": "...", ...}, ...]}
                places_list = radius_data.get("found", [])
                place_ids = [place.get("xid") for place in places_list if place.get("xid")]
            elif "features" in radius_data:
                # GeoJSON format
                for feature in radius_data.get("features", []):
                    if "properties" in feature:
                        props = feature["properties"]
                        if props.get("xid"):
                            place_ids.append(props["xid"])
                            places_list.append(props)
            elif "data" in radius_data:
                # Alternative format
                places_list = radius_data.get("data", [])
                place_ids = [place.get("xid") for place in places_list if isinstance(place, dict) and place.get("xid")]
        elif isinstance(radius_data, list):
            # Direct list format
            places_list = radius_data
            place_ids = [place.get("xid") for place in places_list if isinstance(place, dict) and place.get("xid")]

        logging.info(f"Found {len(places_list)} places in response, {len(place_ids)} with xids")

        # Try to extract data directly from radius response first (faster, no extra API calls)
        stops = []
        for place in places_list[:10]:
            try:
                # Check if place has name directly in response
                name = place.get("name") or place.get("name:en") or place.get("name:en:1")
                if not name and "properties" in place:
                    props = place["properties"]
                    name = props.get("name") or props.get("name:en") or props.get("name:en:1")
                
                if name:
                    # Extract type from kinds
                    kinds = place.get("kinds", "") or place.get("properties", {}).get("kinds", "")
                    if kinds:
                        kind_list = [k.strip().replace("_", " ").title() for k in kinds.split(",") if k.strip() and not k.startswith("other")]
                        kind_type = kind_list[0] if kind_list else "Attraction"
                    else:
                        kind_type = "Attraction"
                    
                    # Get location
                    address = place.get("address", {}) or place.get("properties", {}).get("address", {})
                    location_name = address.get("city") or address.get("town") or address.get("village") or drop
                    
                    stops.append({
                        "name": name,
                        "type": kind_type,
                        "location": location_name,
                        "distance": "Near destination"
                    })
            except Exception as e:
                logging.warning(f"Error processing place from radius response: {e}")
                continue

        # If we didn't get enough places from radius response, fetch details using xid
        if len(stops) < 5 and place_ids:
            logging.info(f"Only got {len(stops)} places from radius, fetching details for {len(place_ids)} places...")
            otm_details_url = "https://api.opentripmap.com/0.1/en/places/xid"
            
            for xid in place_ids[:8]:
                if len(stops) >= 5:
                    break
                    
                try:
                    details_params = {"apikey": otm_key}
                    details_response = requests.get(f"{otm_details_url}/{xid}", params=details_params, timeout=10)
                    
                    if details_response.status_code != 200:
                        logging.warning(f"Failed to get details for place {xid}: {details_response.status_code}")
                        continue

                    place_data = details_response.json()
                    
                    # Extract place information
                    name = place_data.get("name") or place_data.get("name:en") or place_data.get("name:en:1")
                    if not name:
                        continue
                    
                    # Check if we already have this place
                    if any(s["name"] == name for s in stops):
                        continue
                    
                    # Get place type from kinds
                    kinds = place_data.get("kinds", "")
                    if kinds:
                        kind_list = [k.strip().replace("_", " ").title() for k in kinds.split(",") if k.strip() and not k.startswith("other")]
                        kind_type = kind_list[0] if kind_list else "Attraction"
                    else:
                        kind_type = "Attraction"
                    
                    # Get address/location
                    address = place_data.get("address", {})
                    location_name = address.get("city") or address.get("town") or address.get("village") or drop
                    
                    stops.append({
                        "name": name,
                        "type": kind_type,
                        "location": location_name,
                        "distance": "Near destination"
                    })
                    
                except Exception as place_error:
                    logging.warning(f"Error fetching details for place {xid}: {place_error}")
                    continue

        if not stops:
            logging.warning(f"No tourist places found from OpenTripMap for {drop}, trying Overpass API fallback...")
            # Fallback: Use Overpass API to get tourist attractions
            try:
                overpass_url = "http://overpass-api.de/api/interpreter"
                # Expanded query to get more types of tourist places
                query = f"""
                [out:json][timeout:25];
                (
                  node["tourism"~"^(attraction|museum|gallery|theme_park|zoo|monument|viewpoint|artwork|information|picnic_site|wilderness_hut)$"](around:20000,{dest_lat},{dest_lon});
                  way["tourism"~"^(attraction|museum|gallery|theme_park|zoo|monument|viewpoint|artwork|information|picnic_site|wilderness_hut)$"](around:20000,{dest_lat},{dest_lon});
                  node["historic"](around:20000,{dest_lat},{dest_lon});
                  way["historic"](around:20000,{dest_lat},{dest_lon});
                  node["leisure"~"^(park|nature_reserve|garden)$"](around:20000,{dest_lat},{dest_lon});
                  way["leisure"~"^(park|nature_reserve|garden)$"](around:20000,{dest_lat},{dest_lon});
                );
                out body;
                >;
                out skel qt;
                """
                overpass_resp = requests.get(overpass_url, params={"data": query}, timeout=25)
                if overpass_resp.status_code == 200:
                    overpass_data = overpass_resp.json()
                    elements = overpass_data.get("elements", [])
                    logging.info(f"Overpass API returned {len(elements)} elements")
                    for elem in elements[:15]:
                        tags = elem.get("tags", {})
                        name = tags.get("name") or tags.get("name:en") or tags.get("name:hi")
                        if name:
                            # Determine type from tourism or historic tag
                            tourism_type = tags.get("tourism", "")
                            historic_type = tags.get("historic", "")
                            leisure_type = tags.get("leisure", "")
                            
                            if tourism_type:
                                place_type = tourism_type.replace("_", " ").title()
                            elif historic_type:
                                place_type = historic_type.replace("_", " ").title()
                            elif leisure_type:
                                place_type = leisure_type.replace("_", " ").title()
                            else:
                                place_type = "Attraction"
                            
                            stops.append({
                                "name": name,
                                "type": place_type,
                                "location": drop,
                                "distance": "Near destination"
                            })
                    if stops:
                        logging.info(f"Found {len(stops)} places using Overpass API fallback")
                    else:
                        logging.warning(f"Overpass API returned {len(elements)} elements but none had names")
                else:
                    logging.warning(f"Overpass API returned status {overpass_resp.status_code}")
            except Exception as fallback_error:
                logging.warning(f"Overpass API fallback also failed: {fallback_error}", exc_info=True)

        if not stops:
            logging.error(f"Could not find any tourist places for {drop} using any method")
            return []

        logging.info(f"Successfully found {len(stops)} real tourist places for {drop}")
        return stops[:5]  # Return top 5 places

    except Exception as e:
        logging.error(f"Error in generate_itinerary: {e}", exc_info=True)
        # Try Overpass API as last resort
        try:
            geo_url = "https://nominatim.openstreetmap.org/search"
            headers = {"User-Agent": "TripNinjaAgent/1.0"}
            drop_resp = requests.get(geo_url, params={"q": drop, "format": "json", "limit": 1}, headers=headers, timeout=10)
            if drop_resp.status_code == 200:
                drop_data = drop_resp.json()
                if drop_data:
                    dest_lat = float(drop_data[0]['lat'])
                    dest_lon = float(drop_data[0]['lon'])
                    overpass_url = "http://overpass-api.de/api/interpreter"
                    query = f"""
                    [out:json][timeout:25];
                    (
                      node["tourism"~"^(attraction|museum|gallery|theme_park|zoo|monument|viewpoint|artwork)$"](around:10000,{dest_lat},{dest_lon});
                      way["tourism"~"^(attraction|museum|gallery|theme_park|zoo|monument|viewpoint|artwork)$"](around:10000,{dest_lat},{dest_lon});
                    );
                    out body;
                    >;
                    out skel qt;
                    """
                    overpass_resp = requests.get(overpass_url, params={"data": query}, timeout=20)
                    if overpass_resp.status_code == 200:
                        overpass_data = overpass_resp.json()
                        elements = overpass_data.get("elements", [])
                        stops = []
                        for elem in elements[:5]:
                            tags = elem.get("tags", {})
                            name = tags.get("name") or tags.get("name:en")
                            if name:
                                tourism_type = tags.get("tourism", "attraction").replace("_", " ").title()
                                stops.append({
                                    "name": name,
                                    "type": tourism_type,
                                    "location": drop,
                                    "distance": "Near destination"
                                })
                        if stops:
                            return stops[:5]
        except:
            pass
        return []  # Return empty array if all methods fail

@tool
def get_hotels(city: str, room_type: str, nights: int, budget: float) -> List[Dict[str, Any]]:
    """Fetch real hotels dynamically using Overpass API from OpenStreetMap"""
    try:
        # Get city coordinates
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "TripNinjaAgent/1.0"}
        city_resp = requests.get(geo_url, params={"q": city, "format": "json", "limit": 1}, headers=headers, timeout=10)
        
        if city_resp.status_code != 200 or not city_resp.json():
            logging.warning(f"Could not geocode city: {city}")
            return []
        
        city_data = city_resp.json()
        lat, lon = float(city_data[0]['lat']), float(city_data[0]['lon'])
        
        # Query Overpass API for hotels and accommodations
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json][timeout:25];
        (
          node["tourism"="hotel"](around:10000,{lat},{lon});
          node["tourism"="hostel"](around:10000,{lat},{lon});
          node["tourism"="guest_house"](around:10000,{lat},{lon});
          node["tourism"="resort"](around:10000,{lat},{lon});
          way["tourism"="hotel"](around:10000,{lat},{lon});
          way["tourism"="hostel"](around:10000,{lat},{lon});
          way["tourism"="guest_house"](around:10000,{lat},{lon});
          way["tourism"="resort"](around:10000,{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        overpass_resp = requests.get(overpass_url, params={"data": query}, timeout=20)
        
        if overpass_resp.status_code != 200:
            logging.warning(f"Overpass API failed for hotels: {overpass_resp.status_code}")
            return []
        
        overpass_data = overpass_resp.json()
        elements = overpass_data.get("elements", [])
        
        hotels = []
        base_prices = {
            "single": 1500,
            "double": 2500,
            "deluxe": 4000
        }
        base_price = base_prices.get(room_type.lower(), 2500)
        
        # Common amenities
        all_amenities = ["WiFi", "Breakfast", "AC", "Parking", "Pool", "Gym", "Spa", "Restaurant"]
        
        for elem in elements[:15]:  # Get up to 15 hotels
            tags = elem.get("tags", {})
            name = tags.get("name") or tags.get("name:en")
            
            if not name:
                continue
            
            # Determine hotel type/rating based on tourism tag
            tourism_type = tags.get("tourism", "hotel")
            if tourism_type == "resort":
                rating = 4.5
                price_multiplier = 1.5
                amenities = all_amenities[:6]  # More amenities for resorts
            elif tourism_type == "hotel":
                rating = 4.2
                price_multiplier = 1.0
                amenities = all_amenities[:4]  # Standard amenities
            elif tourism_type == "guest_house":
                rating = 4.0
                price_multiplier = 0.7
                amenities = all_amenities[:3]  # Basic amenities
            else:  # hostel
                rating = 3.8
                price_multiplier = 0.5
                amenities = all_amenities[:2]  # Minimal amenities
            
            # Calculate price based on room type and hotel category
            price = int(base_price * price_multiplier)
            
            # Adjust price based on room type
            if room_type.lower() == "single":
                price = int(price * 0.8)
            elif room_type.lower() == "deluxe":
                price = int(price * 1.3)
            
            # Add some variation to prices
            price = int(price * random.uniform(0.9, 1.1))
            
            hotels.append({
                "name": name,
                "type": room_type.capitalize(),
                "price": price,
                "rating": round(rating + random.uniform(-0.2, 0.2), 1),
                "amenities": amenities
            })
        
        # Filter by budget (total for all nights)
        max_price_per_night = budget / nights if nights > 0 else budget
        filtered = [h for h in hotels if h["price"] <= max_price_per_night]
        
        # Sort by rating (highest first)
        filtered.sort(key=lambda x: x["rating"], reverse=True)
        
        logging.info(f"Found {len(filtered)} hotels in {city} within budget")
        return filtered[:3]  # Return top 3 hotels
    
    except Exception as e:
        logging.error(f"Error in get_hotels: {e}", exc_info=True)
        return []

@tool
def get_hospitals(city: str) -> List[str]:
    """Fetch hospitals dynamically using Overpass API"""
    try:
        geo_url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": "TripNinjaAgent/1.0"}
        city_resp = requests.get(geo_url, params={"q": city, "format": "json"}, headers=headers, timeout=10).json()
        
        if not city_resp:
            return []
        
        lat, lon = city_resp[0]['lat'], city_resp[0]['lon']

        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f"""
        [out:json];
        node
          ["amenity"="hospital"]
          (around:10000,{lat},{lon});
        out;
        """
        r = requests.get(overpass_url, params={"data": query}, timeout=15).json()
        hospitals = [elem["tags"]["name"] for elem in r.get("elements", []) if "name" in elem.get("tags", {})]

        return hospitals[:4] if hospitals else ["City Hospital", "General Medical Center"]

    except Exception as e:
        logging.error(f"Error in get_hospitals: {e}")
        return ["City Hospital", "General Medical Center"]

# ============================================================================
# AGENTIC SYSTEM SETUP
# ============================================================================

# Collect all tools
trip_planning_tools = [
    search_rides,
    get_weather,
    generate_itinerary,
    get_hotels,
    get_hospitals
]

# Initialize the agent
_agent_executor = None

def get_agent_executor():
    """Initialize and return the agent executor (singleton pattern)"""
    global _agent_executor
    if _agent_executor is not None:
        return _agent_executor
    
    try:
        # Get Google API key
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            logging.warning("GOOGLE_API_KEY not found. Agent will use fallback mode.")
            return None
        
        # Initialize Google GenAI model
        # Try different model name formats to find one that works
        model_names = [
            "models/gemini-pro",  # Full model path format
            "gemini-pro",          # Short name
            "models/gemini-1.5-pro",  # Alternative model
            "gemini-1.5-pro",
            "models/gemini-1.0-pro",  # Older version
            "gemini-1.0-pro"
        ]
        
        llm = None
        last_error = None
        
        for model_name in model_names:
            try:
                logging.info(f"Trying model: {model_name}")
                llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=google_api_key,
                    temperature=0.1
                )
                # Model initialized successfully (will be validated on first use)
                logging.info(f"Successfully initialized model: {model_name}")
                break
            except Exception as e:
                last_error = e
                logging.warning(f"Model {model_name} failed: {str(e)[:200]}")
                continue
        
        if llm is None:
            logging.error(f"Failed to initialize any Gemini model. Last error: {last_error}")
            logging.error("Please check your GOOGLE_API_KEY and ensure it has access to Gemini models")
            return None
        
        # Create system prompt for the agent
        system_prompt = """You are TripNinja, an intelligent travel planning agent. Your role is to help users plan complete trips by coordinating multiple tools.

When a user requests a trip plan, you should:
1. Use search_rides to find transportation options from pickup to drop location
2. Use get_weather to check weather at the destination
3. Use generate_itinerary to find tourist attractions at the destination
4. Use get_hospitals to find nearby hospitals for safety
5. If bookAccommodation is True, use get_hotels to find hotels

Always use the exact parameters provided by the user. Return a complete JSON response with all the data collected from the tools.

Your response must be a valid JSON object with these keys:
- rides: array of ride options
- weather: object with weather data
- hotels: array of hotels (empty if bookAccommodation is False)
- itinerary: array of tourist attractions
- hospitals: array of hospital names

Be thorough and use all relevant tools to provide comprehensive trip planning information."""

        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create tool-calling agent (LangChain 1.0+)
        agent = create_tool_calling_agent(llm, trip_planning_tools, prompt)
        
        # Create agent executor
        _agent_executor = AgentExecutor(
            agent=agent,
            tools=trip_planning_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
        
        logging.info("Agent executor initialized successfully")
        return _agent_executor
        
    except Exception as e:
        logging.error(f"Error initializing agent: {e}", exc_info=True)
        return None

# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks

@api_router.post("/plan-trip")
async def plan_trip(request: TripRequest):
    """Main endpoint to plan a complete trip using agentic system"""
    try:
        # Get agent executor
        agent_executor = get_agent_executor()
        
        # If agent is not available, fall back to direct function calls
        if agent_executor is None:
            logging.info("Agent not available, using direct function calls as fallback")
            rides = search_rides.invoke({
                "pickup": request.pickup,
                "drop": request.drop,
                "budget": request.budget,
                "vehicle_type": request.vehicleType
            })
            weather = get_weather.invoke({"location": request.drop})
            hotels = []
            if request.bookAccommodation:
                hotels = get_hotels.invoke({
                    "city": request.drop,
                    "room_type": request.roomType,
                    "nights": request.nights,
                    "budget": request.budget
                })
            itinerary = generate_itinerary.invoke({
                "pickup": request.pickup,
                "drop": request.drop
            })
            hospitals = get_hospitals.invoke({"city": request.drop})
            
            return {
                "rides": rides,
                "weather": weather,
                "hotels": hotels,
                "itinerary": itinerary,
                "hospitals": hospitals
            }
        
        # Construct agent prompt with all request parameters
        agent_input = f"""Plan a trip with the following details:
- Pickup location: {request.pickup}
- Drop/destination: {request.drop}
- Budget: ₹{request.budget}
- Vehicle type: {request.vehicleType}
- Number of nights: {request.nights}
- Room type: {request.roomType}
- Book accommodation: {request.bookAccommodation}
- Departure date: {request.departDate or 'Not specified'}

Please use the available tools to:
1. Search for rides from {request.pickup} to {request.drop} with budget ₹{request.budget} and vehicle type {request.vehicleType}
2. Get weather information for {request.drop}
3. Generate a tourist itinerary for attractions near {request.drop}
4. Find nearby hospitals in {request.drop}
{"5. Find hotels in " + request.drop + " with room type " + request.roomType + " for " + str(request.nights) + " nights within budget ₹" + str(request.budget) if request.bookAccommodation else ""}

Return a JSON response with all the collected data in this exact format:
{{
    "rides": [...],
    "weather": {{...}},
    "hotels": [...],
    "itinerary": [...],
    "hospitals": [...]
}}"""
        
        # Execute agent
        logging.info("Executing agent for trip planning...")
        try:
            result = agent_executor.invoke({"input": agent_input})
        except Exception as agent_error:
            # If agent execution fails (e.g., model not available), fall back to direct calls
            logging.warning(f"Agent execution failed: {agent_error}. Falling back to direct function calls.")
            rides = search_rides.invoke({
                "pickup": request.pickup,
                "drop": request.drop,
                "budget": request.budget,
                "vehicle_type": request.vehicleType
            })
            weather = get_weather.invoke({"location": request.drop})
            hotels = []
            if request.bookAccommodation:
                hotels = get_hotels.invoke({
                    "city": request.drop,
                    "room_type": request.roomType,
                    "nights": request.nights,
                    "budget": request.budget
                })
            itinerary = generate_itinerary.invoke({
                "pickup": request.pickup,
                "drop": request.drop
            })
            hospitals = get_hospitals.invoke({"city": request.drop})
            
            return {
                "rides": rides,
                "weather": weather,
                "hotels": hotels,
                "itinerary": itinerary,
                "hospitals": hospitals
            }
        
        # Extract data from agent's tool calls
        intermediate_steps = result.get("intermediate_steps", [])
        
        # Initialize response structure
        response_data = {
            "rides": [],
            "weather": {},
            "hotels": [],
            "itinerary": [],
            "hospitals": []
        }
        
        # Extract results from tool calls
        for step in intermediate_steps:
            if len(step) >= 2:
                # step[0] is the AgentAction, step[1] is the tool result
                agent_action = step[0]
                tool_result = step[1]
                
                # Get tool name from agent action
                tool_name = ""
                if hasattr(agent_action, 'tool'):
                    tool_name = agent_action.tool
                elif hasattr(agent_action, 'tool_input'):
                    # Try to infer from tool_input or other attributes
                    tool_name = str(agent_action)
                
                # Map tool results to response structure
                tool_name_lower = tool_name.lower()
                if "search_rides" in tool_name_lower or "rides" in tool_name_lower:
                    if isinstance(tool_result, list):
                        response_data["rides"] = tool_result
                    elif isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, list):
                                response_data["rides"] = parsed
                        except:
                            pass
                elif "get_weather" in tool_name_lower or "weather" in tool_name_lower:
                    if isinstance(tool_result, dict):
                        response_data["weather"] = tool_result
                    elif isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, dict):
                                response_data["weather"] = parsed
                        except:
                            pass
                elif "get_hotels" in tool_name_lower or "hotels" in tool_name_lower:
                    if isinstance(tool_result, list):
                        response_data["hotels"] = tool_result
                    elif isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, list):
                                response_data["hotels"] = parsed
                        except:
                            pass
                elif "generate_itinerary" in tool_name_lower or "itinerary" in tool_name_lower:
                    if isinstance(tool_result, list):
                        response_data["itinerary"] = tool_result
                    elif isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, list):
                                response_data["itinerary"] = parsed
                        except:
                            pass
                elif "get_hospitals" in tool_name_lower or "hospitals" in tool_name_lower:
                    if isinstance(tool_result, list):
                        response_data["hospitals"] = tool_result
                    elif isinstance(tool_result, str):
                        try:
                            parsed = json.loads(tool_result)
                            if isinstance(parsed, list):
                                response_data["hospitals"] = parsed
                        except:
                            pass
        
        # If agent didn't call all tools, fill in missing data
        if not response_data["rides"]:
            response_data["rides"] = search_rides.invoke({
                "pickup": request.pickup,
                "drop": request.drop,
                "budget": request.budget,
                "vehicle_type": request.vehicleType
            })
        if not response_data["weather"]:
            response_data["weather"] = get_weather.invoke({"location": request.drop})
        if not response_data["itinerary"]:
            response_data["itinerary"] = generate_itinerary.invoke({
                "pickup": request.pickup,
                "drop": request.drop
            })
        if not response_data["hospitals"]:
            response_data["hospitals"] = get_hospitals.invoke({"city": request.drop})
        if request.bookAccommodation and not response_data["hotels"]:
            response_data["hotels"] = get_hotels.invoke({
                "city": request.drop,
                "room_type": request.roomType,
                "nights": request.nights,
                "budget": request.budget
            })
        
        logging.info(f"Agent completed trip planning. Found {len(response_data['rides'])} rides, {len(response_data['itinerary'])} attractions, {len(response_data['hospitals'])} hospitals")
        return response_data
    
    except Exception as e:
        logging.error(f"Error in plan_trip: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Include the router in the main app
app.include_router(api_router)

# This middleware will now read from your .env file
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. REMOVED THE OLD SHUTDOWN EVENT
# @app.on_event("shutdown")
# async def shutdown_db_client():
#     client.close()