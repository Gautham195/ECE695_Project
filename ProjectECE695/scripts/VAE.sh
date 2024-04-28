'''VAE Training'''

epochs=100
learning_rate=0.001
batch_size=32
dataset=SMAP
sequence_length=200
feature_size=1
latent_size=70
hidden_size=90
channel=A-3

python VAE.py --n_epochs ${epochs} --batch_size ${batch_size} --lr ${learning_rate} --sequence_length ${sequence_length} --latent_dim ${latent_size} --feature_size ${feature_size} --hidden_size ${hidden_size} --channel ${channel} --dataset ${dataset}