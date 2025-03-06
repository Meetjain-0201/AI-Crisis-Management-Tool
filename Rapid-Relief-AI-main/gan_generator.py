import tensorflow as tf
import numpy as np
from pymongo import MongoClient
import time
from datetime import datetime

class GANGenerator:
    def __init__(self):
        # Initialize MongoDB connection
        try:
            self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
            self.db = self.client["resource_allocation"]
            self.collection = self.db["gan_data"]
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            self.client = None
            self.db = None
            self.collection = None
        
        # Generator Network
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='leaky_relu', input_shape=(5,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='leaky_relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15)  # 5 cities * 3 metrics each
        ])
        
        # Discriminator Network
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='leaky_relu', input_shape=(15,)),
            tf.keras.layers.Dense(32, activation='leaky_relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Combined Network
        self.combined = tf.keras.Sequential([self.generator, self.discriminator])
        
        self.city_templates = {
            "Delhi": {"base_population": 250000, "base_resources": {"food": 2000, "water": 3000, "medical": 1000}},
            "Mumbai": {"base_population": 300000, "base_resources": {"food": 2500, "water": 3500, "medical": 1200}},
            "Chennai": {"base_population": 200000, "base_resources": {"food": 1800, "water": 2800, "medical": 900}},
            "Hyderabad": {"base_population": 180000, "base_resources": {"food": 1600, "water": 2600, "medical": 800}},
            "Bangalore": {"base_population": 220000, "base_resources": {"food": 2000, "water": 3000, "medical": 1000}}
        }

    def generate(self):
            
        noise = np.random.normal(0, 1, (1, 5))
        gan_output = self.generator.predict(noise, verbose=0)[0]
        
        synthetic_data = []
        for idx in range(5):  # 5 cities
            city_name = list(self.city_templates.keys())[idx]
            base_pop = self.city_templates[city_name]["base_population"]
            
            data_point = {
                "region_id": idx,
                "region_name": city_name,
                "population_density": int(abs(gan_output[idx*3] * 50000) + base_pop),
                "road_block_status": int(abs(gan_output[idx*3 + 1] * 5)),
                "severity_score": float(abs(gan_output[idx*3 + 2] * 100)),
                "warehouse_stock_status": self._generate_stocks(city_name),
                "resource_needs": self._generate_needs(city_name),
                "timestamp": datetime.now()
            }
            synthetic_data.append(data_point)
        
        return synthetic_data

    def _generate_stocks(self, city_name):
        base = self.city_templates[city_name]["base_resources"]
        return {k: v * np.random.uniform(0.5, 1.5) for k, v in base.items()}

    def _generate_needs(self, city_name):
        base = self.city_templates[city_name]["base_resources"]
        return {k: v * np.random.uniform(0.8, 2.0) for k, v in base.items()}

    def run(self):
        while True:
            try:
                if self.collection is None:
                    print("No database connection")
                    time.sleep(5)
                    continue
                    
                data = self.generate()
                self.collection.delete_many({})
                self.collection.insert_many(data)
                time.sleep(3)
            except Exception as e:
                print(f"Error in run loop: {e}")
                time.sleep(5)

if __name__ == "__main__":
    gan = GANGenerator()
    gan.run()
