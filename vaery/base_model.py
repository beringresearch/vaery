import tensorflow as tf
tf.random.set_seed(1234)

def BaseModel(input_shape=32, latent_dim=1, hidden_nodes_1=512, hidden_nodes_2=64):
    input_encoder = tf.keras.layers.Input(shape=(input_shape,), name="Input_Encoder")
    batch_normalize1 = tf.keras.layers.BatchNormalization()(input_encoder)
    hidden_layer1 = tf.keras.layers.Dense(hidden_nodes_1, activation="relu", name="Hidden_Encoding1")(batch_normalize1)
    batch_normalize2 = tf.keras.layers.BatchNormalization()(hidden_layer1)
    hidden_layer2 = tf.keras.layers.Dense(hidden_nodes_2, activation="relu", name="Hidden_Encoding2")(batch_normalize2)
    batch_normalize3 = tf.keras.layers.BatchNormalization()(hidden_layer2)
    z = tf.keras.layers.Dense(latent_dim, name="Mean")(batch_normalize3)
    
    encoder = tf.keras.Model(input_encoder, z)
    
    input_decoder = tf.keras.layers.Input(shape=(latent_dim,), name="Input_Decoder")
    batch_normalize1 = tf.keras.layers.BatchNormalization()(input_decoder)
    decoder_hidden_layer1 = tf.keras.layers.Dense(hidden_nodes_2, activation="relu", name="Hidden_Decoding1")(batch_normalize1)
    batch_normalize2 = tf.keras.layers.BatchNormalization()(decoder_hidden_layer1)
    decoder_hidden_layer2 = tf.keras.layers.Dense(hidden_nodes_1, activation="relu", name="Hidden_Decoding2")(batch_normalize2)
    batch_normalize3 = tf.keras.layers.BatchNormalization()(decoder_hidden_layer2)
    decoded = tf.keras.layers.Dense(input_shape, activation="sigmoid", name="Decoded")(batch_normalize3)
    
    decoder = tf.keras.Model(input_decoder, decoded, name="Decoder")
    
    encoder_decoder = decoder(encoder(input_encoder))

    ae = tf.keras.Model(input_encoder, encoder_decoder)
    
    ae.compile(loss="mean_squared_error", optimizer=tf.optimizers.Adam(lr=1e-4))
    
    return ae

def SimpleBaseModel(input_shape=5, latent_dim=1, hidden_nodes_1=16):
    input_encoder = tf.keras.layers.Input(shape=(input_shape,), name="Input_Encoder")
    batch_normalize1 = tf.keras.layers.BatchNormalization()(input_encoder)
    hidden_layer1 = tf.keras.layers.Dense(hidden_nodes_1, activation="relu", name="Hidden_Encoding1")(batch_normalize1)
    batch_normalize2 = tf.keras.layers.BatchNormalization()(hidden_layer1)
    z = tf.keras.layers.Dense(latent_dim, name="Mean")(batch_normalize2)
    
    encoder = tf.keras.Model(input_encoder, z, name="Encoder")
    
    input_decoder = tf.keras.layers.Input(shape=(latent_dim,), name="Input_Decoder")
    batch_normalize1 = tf.keras.layers.BatchNormalization()(input_decoder)
    decoder_hidden_layer1 = tf.keras.layers.Dense(hidden_nodes_1, activation="relu", name="Hidden_Decoding1")(batch_normalize1)
    batch_normalize2 = tf.keras.layers.BatchNormalization()(decoder_hidden_layer1)
    decoded = tf.keras.layers.Dense(input_shape, activation="linear", name="Decoded")(batch_normalize2)
    
    decoder = tf.keras.Model(input_decoder, decoded, name="Decoder")
    
    encoder_decoder = decoder(encoder(input_encoder))

    ae = tf.keras.Model(input_encoder, encoder_decoder)
    
    ae.compile(loss="mean_squared_error", optimizer=tf.optimizers.Adam())
    
    return ae