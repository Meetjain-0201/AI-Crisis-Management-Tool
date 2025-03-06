class GANGenerator:
    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        self.db = self.client["resource_allocation"]
        self.collection = self.db["gan_data"]
        
        # Improved Generator
        self.generator = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='leaky_relu', input_shape=(5,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(64, activation='leaky_relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(15, activation='tanh')
        ])
        
        # Added Discriminator
        self.discriminator = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='leaky_relu', input_shape=(15,)),
            tf.keras.layers.Dense(32, activation='leaky_relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Connection retry logic
        self._ensure_connection()
    
    def _ensure_connection(self):
        try:
            self.client.server_info()
        except Exception as e:
            print(f"Database connection error: {e}")
            self.client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
    
    def generate(self):
        try:
            noise = np.random.normal(0, 1, (1, 5))
            gan_output = self.generator.predict(noise, verbose=0)[0]
            
            synthetic_data = []
            for idx in range(5):
                city_name = list(self.city_templates.keys())[idx]
                base_pop = self.city_templates[city_name]["base_population"]
                
                # Time-based scaling
                current_hour = datetime.now().hour
                time_factor = 1 + (0.2 * np.sin(current_hour * np.pi / 12))
                
                data_point = {
                    "region_id": idx,
                    "region_name": city_name,
                    "population_density": int(abs(gan_output[idx*3] * 50000 * time_factor) + base_pop),
                    "road_block_status": int(abs(gan_output[idx*3 + 1] * 5)),
                    "severity_score": float(np.clip(abs(gan_output[idx*3 + 2] * 100), 0, 100)),
                    "warehouse_stock_status": self._generate_stocks(city_name, gan_output),
                    "resource_needs": self._generate_needs(city_name, gan_output),
                    "timestamp": datetime.now()
                }
                synthetic_data.append(data_point)
            
            return synthetic_data
        except Exception as e:
            print(f"Generation error: {e}")
            return []

    def _generate_stocks(self, city_name, gan_output):
        base = self.city_templates[city_name]["base_resources"]
        idx = list(self.city_templates.keys()).index(city_name)
        noise_factor = gan_output[idx*3 + 2]  # Use GAN output for variability
        return {k: v * np.clip(np.random.uniform(0.5, 1.5) + (0.2 * noise_factor), 0.3, 2.0) for k, v in base.items()}
