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
    
    # Added adversarial training
    gan.discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    gan.combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
    
    # Training with callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('gan_weights.h5', save_best_only=True)
    ]
    
    # Improved training loop with discriminator
    for epoch in range(100):
        # Train discriminator
        idx = np.random.randint(0, X.shape[0], 32)
        real_data = Y[idx]
        fake_data = gan.generator.predict(X[idx], verbose=0)
        d_loss_real = gan.discriminator.train_on_batch(real_data, np.ones((32, 1)))
        d_loss_fake = gan.discriminator.train_on_batch(fake_data, np.zeros((32, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        
        # Train generator
        g_loss = gan.combined.train_on_batch(X[idx], np.ones((32, 1)))
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | D Loss: {d_loss[0]:.4f} | G Loss: {g_loss:.4f}")
    
    print("Training completed. Weights saved.")

if __name__ == "__main__":
    tf.get_logger().setLevel('ERROR')
    quick_train()
