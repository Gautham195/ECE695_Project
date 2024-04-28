from torch import  nn
import torch
import torch.nn.functional as F

'''
class Encoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=2, batch_first=True)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_sigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h_n = h_n[-1]  # Consider only the last layer because the final layer has all the data about the past
        mu = self.fc_mu(h_n)
        sigma = self.fc_sigma(h_n)
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, latent_size, feature_size):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.lstm = nn.LSTM(hidden_size, feature_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(latent_size, sequence_length * hidden_size)
        #self.fc_sigma = nn.Linear(hidden_size, latent_size)
        
    def forward(self, z):
        z = self.fc(z)
        z = z.view(-1, self.sequence_length, self.hidden_size)  # Add a dimension for sequence length
        output, _ = self.lstm(z)
        return output

'''
class Encoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.intermediate_hidden_size = hidden_size * 2
        self.linear1 = nn.Linear(sequence_length * feature_size, self.intermediate_hidden_size)
        self.linear2 = nn.Linear(self.intermediate_hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, latent_size)
        self.fc_sigma = nn.Linear(hidden_size, latent_size)

    def forward(self, x):
        x = x.view(-1, self.sequence_length * self.feature_size)
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        mu = self.fc_mu(x)
        sigma = self.fc_sigma(x)
        return mu, sigma
    
class Decoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, latent_size, feature_size):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.feature_size = feature_size
        self.intermediate_hidden_size = hidden_size * 2
        self.linear1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
        self.linear2 = nn.Linear(self.intermediate_hidden_size, sequence_length * feature_size)
        self.fc = nn.Linear(latent_size, hidden_size)

    def forward(self, z):
        #print(z.shape)
        z = self.fc(z)
        z = torch.relu(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        z = z.view(-1, self.sequence_length, self.feature_size)
        return z

'''Basic Autoencoder. Nothing special'''
class Autoencoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(sequence_length, hidden_size, feature_size, latent_size)
        self.decoder = Decoder(sequence_length, hidden_size, latent_size, feature_size)

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstructed_x = self.decoder(latent_representation)
        return reconstructed_x, latent_representation
    
    

'''Class Encoder:
    Represents the Encoder architecture - 
    input->LSTM1->LSTM2->FC_mu
                       ->FC_log_var
'''
class VAE(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(sequence_length, hidden_size, feature_size, latent_size)
        self.decoder = Decoder(sequence_length, hidden_size, latent_size, feature_size)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, mu, log_var