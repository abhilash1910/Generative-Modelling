import torch

class GPT__MLP(torch.nn.Module):
    def __init__(self,hidden_size,intermediate_size):
        #hidden size,intermediate size (hidden_size=embedding size):
        self.ln1=torch.nn.LayerNorm(hidden_size,intermediate_size)
        self.act=torch.nn.functional.softmax()
        self.ln2=torch.nn.LayerNorm(intermediate_size,hidden_size)
        self.dropout=torch.nn.dropout()

    def forward(self,inputs):
        ln1_output=self.ln1(inputs)
        act_output=self.act(ln1_output)
        ln2_output=self.ln2(act_output)
        drop_output=self.dropout(ln2_output)
        return drop_output

class GPT_Attention(torch.nn.Module):
    def __init__(self,hidden_size,num_heads):
    #hidden_size=embed_size and num_heads
        self.embed_dim=hidden_size
        self.attn_head_dim= self.embed_dim//num_heads
        #qkv
        self.q= torch.nn.Dense(self.embed_dim,self.embed_dim)
        self.k= torch.nn.Dense(self.embed_dim,self.embed_dim)
        self.v= torch.nn.Dense(self.embed_dim,self.embed_dim)

    def MHA(self,attn_output,num_heads):
        #MHA
        mha_attn=attn_output+ (self.attn_head_dim,num_heads)
        #torch permute -> batch size, hidden size, num heads
        for i in range(num_heads):
            mha_attn=attn_output+ self.attn_head_dim
        return mha_attn



    def forward(self,embed):
        #softmax(q.kT)*v -> SHA
        qk_mult=self.q@self.k.transpose(-1,-2)
        act_output=torch.nn.softmax(qk_mult)
        attn_output=act_output@self.v

        return attn_output


class GPTModel(torch.nn.Module):
    # This will be 1 decoder block
    def __init__(self,hidden_size,num_heads,num_layers):
        self.gpt_attn=GPT_Attention(hidden_size,num_heads) #hidden 3072, #num heads 32 -> gpt 3.6
        self.mlp= GPT_MLP(hidden_size,hidden_size)
        self.num_layers=num_layers #30
        self.final_activation=torch.nn.functional.softmax()


    def one_decoder_block(self,inputs):
        attn_module=self.gpt_attn(inputs)
        decoder_output=self.mlp(attn_module)
        return decoder_output

    def forward(self,inputs):
        for _ in range(self.num_layers):
            output_decoder=one_decoder_block(inputs)
        return self.final_activation(output_decoder) #-> output logits from your inputs

#need to have an additional LM head for downstream task
class GPTCausalModel_with RLHF(torch.nn.Module):
    def __init__(self,tokenzied_text,hidden_size,num_heads,num_layers,labels):
        #assign using self.
        self.gpt_model=GPTModel(.....)
        self.linear_layer=torch.nn.Dense(.../same dim as output of your gptmodel)

    def human_generated_logits(self,human_recomm_tokens):
        output_logits_after_human_feedback=self.gpt_model(human_recomm_tokens)
        final_human_feedback_logits=self.linear_layer(output_logits_after_human_feedback)
        return final_human_feedback_logits #you loaded the critic logits

    def forward(self,inputs,human_recomm_tokens):
        #infusion :
        # inputs tokenized  -> gpt is generating -> actor logits
        # human recommendation tokenized text (feedback we are giving) -> gpt is generating logits -> critic logits
        # pass both the actor and critic logits inside your PPO(model)
        #get the loss (combined ppo loss) as the output of your gpt rlhf model
        output_logits=self.gpt_model(inputs)
        final_lm_logits=self.linear_layer(output_logits)
        #GPT model is generating this final_lm_logits -> actor
        final_human_feedback_logits=human_generated_logits(human_recomm_tokens)



        ##PPO loss
        final_output_of_your_rlhfgpt_model=PPO(final_lm_logits#logits_actor in combined_ppo_loss function
        ,final_human_feedback_logits)  #Critic logits as human recomm tokenized input is passsed into gpt model)
        #you design the loss function

        #lm_loss=torch.nn.CrossEntropy(final_lm,final_human_logits)#labels)
        #lm_loss=torch.nn.CrossEntropy(final_lm_logits,labels)
        #return lm_loss,final_lm_logits  # if you are running inference -> final_lm_logits , if you are training lm_loss,final_lm_logits
        return final_output_of_your_rlhfgpt_model




class Actor(torch.nn.Module):
    def __init__(self,dim):
        self.dim=dim
        self.ffn= torch.nn.Dense(self.dim)

    def loss(self,original_logits,model_logits):
        kl_loss=-original_logits*np.log(-original_logits/model_logits)
        return kl_loss

    def forward(self,inputs):
        output_ffn=self.ffn(inputs)
        return loss(inputs,output_ffn)

class Critic(torch.nn.Module):
    def __init__(self,dim):
        self.dim=dim
        self.ffn= torch.nn.Dense(self.dim)

    def loss_ppo(self,original_logits,model_logits):
        if loss=='kl':
            kl_loss=-original_logits*np.log(-original_logits/model_logits)
            return kl_loss
        else:
            return torch.nn.CrossEntropy(original_logits,model_logits).clamp(-threshold,threshold)

    def forward(self,inputs):
        output_ffn=self.ffn(inputs)
        return ppo(inputs,output_ffn)



class PPO (torch.nn.Module):
    # actor first takes the tokenized text as input and predicts its value
    # does loss and then critic also does the tokenized text and predicts its value
    #  now we write a ppo clip loss to compare the actor loss and critic loss(combined ppo loss)
    # next we pass this ppo clip loss as output to the rlhf gpt model
    def __init__(self, tokenized_text):
        self.actor= Actor()
        #self.actor_loss= self.actor.loss() #KL
        self.critic =Critic()
        #self.critic_loss //

    def combined_ppo_loss(self,logits_actor,logits_critic):
        #same as the loss in Critic class
        #critic will update its policy weights
        #actor will update its value weights
        #clipping is important for PPO based methods


    def forward(self,inputs):
        outputs_actor_loss=self.actor(inputs)
        output_critic_loss=self.critic(inputs)
        return combined_ppo_loss(outputs_actor_loss,output_critic_loss) #loss from your tokenized inputs which we provided as feedback to GPT model










