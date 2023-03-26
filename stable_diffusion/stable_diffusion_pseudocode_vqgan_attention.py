
import torch

#diffusion model -> flows -> forward p(x,xt-1)
#backward-> q(xt-1|xt)
#architecture : any simple NN model incouding resnets + some fancy attention mechanism  + the diffusion model loss
# unimodal -> output will be a close reconstructed variant of the input image

import VQGAN
#image model can be anything -> we chose vqgan for this isntance
class Custom_image_attention(torch.nn.Module):
    # attention involving patfhes in input  images similar to the attention mechanism in ViT
    # it divides the inputs into patches of images and then applies the qkv operation that we have seen in the gpt transformer

    def get_patches(self,input_images,patch_simension):
        def divide_into_patches(path_dimension):
            #i-> i+ph , j-> j+ph

            for i in range(input_images.height):
                sum=0

                if(i+ph<=input_images.height):
                    k=i+ph
                for j in range(input_images.width):
                    if(j+pw<=input_images.width):
                        l=j+pw
                        sum+=imput_images[k][l]
                        sum//=patch_dim
                input_images[i][j]=sum

        return divide_into_patches(input_images,patch_dimension)

    #___init__ it would have automatically called
    def initialize(self,inputs_image,num_heads,patch_dimension):
        self.image_embeddings=self.get_patches(inputs_image,patch_dimension)
        self.embed_dims=self.image_embeddings.shape()
        self.num_heads=num_heads
        self.dim_per_attention_head= self.embed_dims//self.num_heads
        self.q= torch.nn.Conv(self.embed_dims,self.embed_dims,initializer=xavier_initializer)
        self.k= torch.nn.Conv(self.embed_dims,self.embed_dims)
        self.v= torch.nn.Conv(self.embed_dims,self.embed_dims)

    def foward(self,input_images):
        qk_mult=self.q@self.k.transpose(-1,-2)
        act_output=torch.nn.softmax(qk_mult)
        attn_output=act_output@self.v

        return attn_output

class Diffusion(torch.nn.Module):
    #you can write diffusion as a loss function or you can choose not to
    def __init__(self,noise_function):
        self.noise_function=torch.normal(scalar_valu=0.6)
        self.vqgan=VQGAN()
        self.Custom_image_attention=Custom_image_attention()

    def sampling(self,input_samples,timesteps,s=0.1):
        alpha_parametr_sampling = torch.cos(((intpu_samples / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_sum_parameter_sampling = alpha_parametr_sampling / alpha_parametr_sampling[0]
        return alpha_parametr_sampling,alpha_sum_parameter_sampling

    def denoising_loss(self,input_logits,model_logits):
        #approximating the loss in backward denoising function q(xt-1|xt) using torch.sqrt -> torch.sqrt()
        #backward denoising funciton loss

        def custom_backward(self,input_logits):
            alpha_parameter_sampling,alpha_sum_parameter_sampling= sampling(input_logits,200)
            posterior_variance_logits = betas * (1. - alpha_parameter_sampling) / (1. - alpha_parameter_sampling)
            return posterior_variance_logits

        return torch.nn.CrossEntropy(posterior_variance_logits,model_logits)



    def forward(self,input_images):
        vqgan_emeddings=self.vqgan(input_images)
        noisy_vqgan_output=self.noise(vqgan_embeddings)
        attented_image=self.Custom_image_attention.initialize(noisy_vqgan_output,num_heads,patch_dimension)
        #forward process of p(xt| xt+1)
        sampled_logits_forward= sampling(attended_image,200)
        input_logits= torch.sqrt((sampled_logits_forward)*(1-sampled_logits_forward))
        #denoising loss
        model_logits=sampled_logits_forward
        return denoising_loss( input_logits,model_logits)











