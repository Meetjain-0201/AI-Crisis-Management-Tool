import numpy as np
from gan_generator import GANGenerator
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def quick_train():
    input_shape = 5
    output_shape = 15
    num_samples = 10000  # Increased dataset size
    
    # Generate correlated features with cluster separation
    X = np.concatenate([
        np.random.normal(0, 1, (num_samples//2, input_shape)),
        np.random.normal(3, 0.5, (num_samples//2, input_shape))
    ])
    Y = np.concatenate([
        np.random.normal(0, 1, (num_samples//2, output_shape)),
        np.random.normal(2, 0.7, (num_samples//2, output_shape))
    ])
    
    gan = GANGenerator()
    
    # Compile networks
    gan.discriminator.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
        metrics=['accuracy']
    )
    
    gan.combined.compile(
        loss='binary_crossentropy', 
        optimizer=tf.keras.optimizers.Adam(0.0002, 0.5)
    )
    
    # Training parameters
    batch_size = 32
    epochs = 100
    
    # Training loop
    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, X.shape[0], batch_size)
        real_data = Y[idx]
        fake_data = gan.generator.predict(X[idx], verbose=0)
        
        d_loss_real = gan.discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
        d_loss_fake = gan.discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        g_loss = gan.combined.train_on_batch(X[idx], np.ones((batch_size, 1)))
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f} | D Acc: {d_loss[1]:.2f}")
    
    # Save weights
    gan.generator.save_weights('gan_generator_weights.h5')
    gan.discriminator.save_weights('gan_discriminator_weights.h5')
    print("Training completed. Weights saved.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    quick_train()
