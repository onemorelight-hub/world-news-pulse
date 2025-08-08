import folium
from geopy.geocoders import Nominatim
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_geo_map(entities):
    try:
        geolocator = Nominatim(user_agent="newspulse_geo")
        m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center on India
        location_counts = Counter([e[0] for e in entities])
        
        for location, count in location_counts.items():
            try:
                geo = geolocator.geocode(location, timeout=5)
                if geo:
                    folium.CircleMarker(
                        location=[geo.latitude, geo.longitude],
                        radius=min(count * 3, 15),  # Smaller radius for optimization
                        popup=f"{location}: {count} mentions",
                        color="blue",
                        fill=True,
                        fill_opacity=0.6
                    ).add_to(m)
            except Exception as e:
                logger.warning(f"Geocoding failed for {location}: {str(e)}")
                continue
        
        return m._repr_html_()
    except Exception as e:
        logger.error(f"Error in create_geo_map: {str(e)}")
        return "<p>Unable to generate map</p>"