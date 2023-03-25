
#imports
class GPT_Transformer():
    def __init__(self,model_name,**kwargs):
        self.tokenizer=AutoTokenizer.from_pretrained()
        self.model=AutoModel.from_pretrained() #AutoModelForCausalLM -> generative

    def forward(text):
        tokens=self.tokenizer(text) #->input_ids,attention_masks,position_ids
        with torch.no_grad():
            output_ids=self.model(**tokens) #-> get logits (logits from outer softmax layers of your GPT LLM)
        #final_logits=output_ids.last_hidden_state
        final_outputs_encoded=torch.nn.Linear(36)(output_ids)
        return final_outputs_encoded

class VQGAN(nn.module):

    def quantizer(image):
        image=image.pad(h,w)
        loss_quantize=0.0
        reconst_quantize=0.0

        #Q
        for i,j in zip(range(h),range(w)):
            x_vec=[0,0,-1,-1,1,1.]
            y_vec=[-1,1,0,1,-1,0.]
            for k,l in zip(x_vec,y_vec):
                u+=x_vec[k]
                v+=y_vec[l]
                loss_quantize+=np.abs(image[i][j]-image[u][v])

        #R
                reconst_quantize+=np.square(np.abs(image[i][j]-image[u][v]))

        #C
                commi_quantize+=np.square(np.abs(image[i][j]-image[u][v]))

        return loss_quantize+ reconst_quantize+commi_quantize

    def Generator(image: List[np.array,torchvision,tf.image]):
        gen_encoder = torch.nn.Conv(filters...)(img.to(device="cuda:0")).to(device="cuda:0")
         #-. add as many as you want
        gen_decoder = torch.nn.Conv(..)(gen_encoder).to(device="cuda:1")
        an_embedd_head_over_your_ged = torch.nn.Conv()(gen_decoder).to(device="cuda:1")

        #vectorquantize function here
        quantized_embeddings= quantizer(an_embed....)
        #loss
        loss_fct=torch.nn.CrossEntropy(an_embedd_head_over_your_ged,labels)
        return loss_fct

    def discriminator(Image):
        #conv layers
        loss_fct=torch.nn.CrossEntropy(....outputfrom your disc convolution,labels)
        return loss_fct

    def codebook_forwards(image):
        #self.create_a_normal_GAN
        def normal_geenrator(iamges):
            //conv blocks
        def normal_discriminator(images):
            //conv blocks
        def loss_normal_gan():
            CrossEntropy()

        self.normal_generator=normal_generator()
        self.normal_disc=normal_disc()
        image=image.to(device="cuda:1")
        self codebook_loss= loss_normal_gan+ lambda*(discriminator.loss.to(device="cuda:1"))

        return self.codebook_loss








    def __init__(self,**kwargs):
        self.generator=Generator()


class CLIP(nn.module):
    self .__init__(self,**kwargs):
        self.config=config

        #self.image_encoder= compvis.load_vqgan_model(config)
        self.image_encoder= VQGAN.self.codebook_loss()
        self.text_encoder= GPT_Transformer.forward(config)


    def projection_head():

    #@forward_ctx
    def forward(self):
        image_encodings= self.image_encoder()
        text_encodings= self.text_encoder()
        #sc+cc
        image_encodings_sc= image_encodings@image_encodings.T/0.2
        text_encodings_sc=text_encodings@text_encodings.T/0.2
        self_encodings_sc = image_encodings+ text_encodings
        #cc
        comb_encodings_cc= text_encodings@image_encodings.T/0.2

        #entropy -plog(p/q)
        kl_entropy = -self_encofings_sc*np.log(self_encofings_sc/comb_envodings_cc)
        return kl_entropy

    def backward():
        @backward_ctx


"""
with tf.gradient_tape():
    x_cap=hjinge_loss(x)
    x_entropy=log(x_cap-x)**2
    return x_entropy
"""