import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Disable eager execution for better performance
tf.compat.v1.disable_eager_execution()

# Load VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Content and style layers
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

@st.cache_data
def load_img(img_file):
    max_dim = 512
    img = Image.open(img_file)
    long = max(img.size)
    scale = max_dim/long
    img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.LANCZOS)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    return img

@st.cache_data
def load_and_process_img(img_file):
    img = load_img(img_file)
    return tf.keras.applications.vgg19.preprocess_input(img)

@st.cache_data
def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3, "Invalid input to deprocessing image"
    
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x

@st.cache_resource
def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    
    return tf.keras.Model(vgg.input, model_outputs)

def get_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a, a, transpose_a=True)
    return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    
    model_outputs = model(init_image)
    
    style_output_features = model_outputs[:num_style_layers]
    content_output_features = model_outputs[num_style_layers:]
    
    style_score = 0
    content_score = 0

    weight_per_style_layer = 1.0 / float(num_style_layers)
    for target_style, comb_style in zip(gram_style_features, style_output_features):
        style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
        
    weight_per_content_layer = 1.0 / float(num_content_layers)
    for target_content, comb_content in zip(content_features, content_output_features):
        content_score += weight_per_content_layer * get_content_loss(comb_content[0], target_content)
    
    style_score *= style_weight
    content_score *= content_weight

    loss = style_score + content_score 
    return loss, style_score, content_score

def compute_loss_and_grads(cfg):
    with tf.GradientTape() as tape: 
        all_loss = compute_loss(**cfg)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, cfg['init_image']), all_loss

@st.cache_data
def run_style_transfer(content_file, 
                       style_file,
                       num_iterations=300,
                       content_weight=1e3, 
                       style_weight=1e-2): 
    model = get_model() 
    for layer in model.layers:
        layer.trainable = False
    
    style_features, content_features = get_feature_representations(model, content_file, style_file)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
    
    init_image = load_and_process_img(content_file)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    
    opt = tf.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)

    best_loss, best_img = float('inf'), None
    
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }
    
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means   
    
    for i in range(num_iterations):
        grads, all_loss = compute_loss_and_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        
        if loss < best_loss:
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % 100 == 0:
            print(f'Iteration: {i}')
    
    return best_img, best_loss

@st.cache_data
def get_feature_representations(model, content_file, style_file):
    content_image = load_and_process_img(content_file)
    style_image = load_and_process_img(style_file)
    
    style_outputs = model(style_image)
    content_outputs = model(content_image)
    
    style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
    content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
    return style_features, content_features

def main():
    st.title("Fast Neural Style Transfer")
    st.write("Upload a content image and a style image to apply style transfer.")

    content_file = st.file_uploader("Choose a content image", type=["jpg", "jpeg", "png"])
    style_file = st.file_uploader("Choose a style image", type=["jpg", "jpeg", "png"])

    if content_file and style_file:
        content_image = Image.open(content_file)
        style_image = Image.open(style_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(content_image, caption="Content Image", use_column_width=True)
        with col2:
            st.image(style_image, caption="Style Image", use_column_width=True)

        if st.button("Apply Style Transfer"):
            with st.spinner("Applying style transfer... This may take a few minutes."):
                best_img, _ = run_style_transfer(content_file, style_file, num_iterations=300)
                st.image(best_img, caption="Style Transfer Result", use_column_width=True)

if __name__ == "__main__":
    main()
