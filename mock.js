// Mock data for TripNinja frontend

export const mockRides = {
  delhi_zirakpur: [
    {
      provider: "Rapido",
      type: "Bike",
      price: 2500,
      eta: "4 hours 30 min",
      distance: "250 km"
    },
    {
      provider: "Ola",
      type: "Auto",
      price: 3500,
      eta: "5 hours",
      distance: "250 km"
    },
    {
      provider: "Uber",
      type: "Car",
      price: 4500,
      eta: "4 hours 15 min",
      distance: "250 km"
    }
  ],
  default: [
    {
      provider: "Rapido",
      type: "Bike",
      price: 1500,
      eta: "2 hours 30 min",
      distance: "150 km"
    },
    {
      provider: "Ola",
      type: "Auto",
      price: 2200,
      eta: "3 hours",
      distance: "150 km"
    },
    {
      provider: "Uber",
      type: "Car",
      price: 3000,
      eta: "2 hours 20 min",
      distance: "150 km"
    }
  ]
};

export const mockWeather = {
  zirakpur: {
    description: "Clear sky",
    temp: 24,
    feelsLike: 22,
    humidity: 45,
    windSpeed: 12
  },
  jaipur: {
    description: "Partly cloudy",
    temp: 28,
    feelsLike: 30,
    humidity: 38,
    windSpeed: 8
  },
  default: {
    description: "Pleasant weather",
    temp: 25,
    feelsLike: 24,
    humidity: 50,
    windSpeed: 10
  }
};

export const mockHotels = {
  zirakpur: [
    {
      name: "Hotel Paradise Inn",
      type: "Double",
      price: 2500,
      rating: 4.2,
      amenities: ["WiFi", "Breakfast", "AC"]
    },
    {
      name: "Green Valley Resort",
      type: "Deluxe",
      price: 3500,
      rating: 4.5,
      amenities: ["WiFi", "Breakfast", "Pool", "AC"]
    }
  ],
  default: [
    {
      name: "City Center Hotel",
      type: "Double",
      price: 2000,
      rating: 4.0,
      amenities: ["WiFi", "Breakfast", "AC"]
    },
    {
      name: "Grand Stay Inn",
      type: "Deluxe",
      price: 3000,
      rating: 4.3,
      amenities: ["WiFi", "Breakfast", "Gym", "AC"]
    }
  ]
};

export const mockItinerary = {
  delhi_zirakpur: [
    {
      name: "Rock Garden",
      type: "Cultural",
      location: "Chandigarh",
      distance: "8 km from Zirakpur"
    },
    {
      name: "Sukhna Lake",
      type: "Nature",
      location: "Chandigarh",
      distance: "12 km from Zirakpur"
    },
    {
      name: "Rose Garden",
      type: "Gardens",
      location: "Chandigarh",
      distance: "10 km from Zirakpur"
    },
    {
      name: "Capitol Complex",
      type: "Historical",
      location: "Chandigarh",
      distance: "11 km from Zirakpur"
    }
  ],
  default: [
    {
      name: "City Museum",
      type: "Cultural",
      location: "Downtown",
      distance: "5 km"
    },
    {
      name: "Central Park",
      type: "Nature",
      location: "City Center",
      distance: "3 km"
    },
    {
      name: "Old Fort",
      type: "Historical",
      location: "Old City",
      distance: "7 km"
    }
  ]
};

export const mockHospitals = {
  zirakpur: [
    "Sohana Hospital",
    "Civil Hospital Zirakpur",
    "Fortis Hospital (Mohali)",
    "Max Super Speciality Hospital"
  ],
  default: [
    "City General Hospital",
    "Metro Medical Center",
    "Community Healthcare"
  ]
};

export const getMockData = (pickup, drop) => {
  const routeKey = `${pickup?.toLowerCase()}_${drop?.toLowerCase()}`.replace(/\s+/g, '_');
  const dropKey = drop?.toLowerCase().replace(/\s+/g, '_');
  
  return {
    rides: mockRides[routeKey] || mockRides.default,
    weather: mockWeather[dropKey] || mockWeather.default,
    hotels: mockHotels[dropKey] || mockHotels.default,
    itinerary: mockItinerary[routeKey] || mockItinerary.default,
    hospitals: mockHospitals[dropKey] || mockHospitals.default
  };
};
