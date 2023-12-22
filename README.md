# Video Diffusion Models

### Abstract
We introduce a novel approach for generating coherent videos using conditional diffusion models. To condition our diffusion model, we use tokens generated from a pre-trained world model.
This integration aims to enhance video generation quality while maintaining consistency over extended periods, representing a technique that could provide new insights into long-form video generation.
Our study conducts a comparative analysis between models conditioned with and without tokens, highlighting the efficacy of each method. The token-conditioned training includes random masking, enhancing the model's robustness and reducing its dependency on tokens. 
We opt for a 2D UNet over a 3D UNet for our diffusion model, motivated by computational considerations and the desire to explore its capabilities in video generation.
Our experimental framework employs a dataset derived from the Atari game, Breakout.
The findings from this research offer insights into advanced conditioning techniques that can bolster consistency in video generation.