from torch import  nn
import torch
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Encoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.latent_size = latent_size
        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.latent_size)


    def forward(self, x):
        # h_0, c_0 = torch.zeros((4, x.shape[0], self.hidden_size), device=x.device), torch.zeros((4, x.shape[0], self.hidden_size), device=x.device) 
        out, (h_n, _) = self.lstm(x) #, (h_0, c_0))
        h_n = h_n[-1]  # Consider only the last layer because the final layer has all the data about the past

        return self.fc(h_n)
    
class Decoder(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Decoder, self).__init__()
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.feature_size = feature_size

        self.lstm = nn.LSTM(feature_size, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, sequence_length)
        
    def forward(self, z):

        output, _ = self.lstm(z.unsqueeze(-1))
        output =  output[:,-1,:].view(-1, self.hidden_size*2)
        return torch.sigmoid(self.fc(output))

'''Basic Autoencoder. Nothing special'''
class Generator(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
        super(Generator, self).__init__()
        self.encoder = Encoder(sequence_length, hidden_size, feature_size, latent_size)
        self.decoder = Decoder(sequence_length, hidden_size, feature_size, latent_size)

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstructed_x = self.decoder(latent_representation)
        return reconstructed_x, latent_representation
    
    

class Discriminator(nn.Module):
    def __init__(self, sequence_length, hidden_size, feature_size):
        super(Discriminator, self).__init__()
        self.sequence_length = sequence_length
        self.feature_size = feature_size
        self.model = nn.Sequential(
            nn.Linear(sequence_length * feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, feature_size),
        )

    def forward(self, x):
        x = x.view(-1, self.sequence_length * self.feature_size)
        return self.model(x)


''' All networks below were used to test '''

# class Encoder_Linear(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Encoder, self).__init__()
#         self.sequence_length = sequence_length
#         self.hidden_size = hidden_size
#         self.feature_size = feature_size
#         self.latent_size = latent_size
#         self.intermediate_hidden_size = hidden_size * 2
#         self.linear1 = nn.Linear(sequence_length * feature_size, self.intermediate_hidden_size)
#         self.linear2 = nn.Linear(self.intermediate_hidden_size, hidden_size)
#         self.fc = nn.Linear(hidden_size, latent_size)

#     def forward(self, x):
#         x = x.view(-1, self.sequence_length * self.feature_size)
#         x = torch.relu(self.linear1(x))
#         x = torch.relu(self.linear2(x))
#         latent_vec = self.fc(x)
#         return latent_vec
    
# class Decoder_Linear(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Decoder, self).__init__()
#         self.sequence_length = sequence_length
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.feature_size = feature_size
#         self.intermediate_hidden_size = hidden_size * 2
#         self.linear1 = nn.Linear(hidden_size, self.intermediate_hidden_size)
#         self.linear2 = nn.Linear(self.intermediate_hidden_size, sequence_length * feature_size)
#         self.fc = nn.Linear(latent_size, hidden_size)

#     def forward(self, z):
#         #print(z.shape)
#         z = self.fc(z)
#         z = torch.relu(self.linear1(z))
#         z = torch.sigmoid(self.linear2(z))
#         z = z.view(-1, self.sequence_length, self.feature_size)
#         return torch.sigmoid(z)


# Encoder network
# class Encoder_Conv(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Encoder, self).__init__()

#         self.sequence_length = sequence_length
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size

#         self.conv1 = nn.Conv1d(in_channels=self.sequence_length * self.feature_size, out_channels=self.hidden_size*2, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=self.hidden_size*2, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=1, stride=2)
#         self.fc1 = nn.Linear(self.hidden_size*3, self.latent_size)
#         self.fc2 = nn.Linear(self.latent_size, self.feature_size)  # Assuming latent space dimension is 100

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten the output from conv layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Decoder network
# class Decoder_Conv(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Decoder, self).__init__()

#         self.sequence_length = sequence_length
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size

#         self.fc1 = nn.Linear(self.feature_size, self.latent_size)
#         self.fc2 = nn.Linear(self.latent_size, self.hidden_size*3)
#         self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1, output_padding=0)
#         self.conv2 = nn.ConvTranspose1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*2, kernel_size=3, stride=1, padding=1, output_padding=0)
#         self.conv3 = nn.ConvTranspose1d(in_channels=self.hidden_size*2, out_channels=self.sequence_length, kernel_size=3, stride=1, padding=1, output_padding=0)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # x = x.view(-1, 128, 7, 7)  # Reshape to match the dimensions before flattening in encoder
#         x = x.view(x.size(0), self.hidden_size*3, self.feature_size)  # Reshape to match the dimensions before flattening in encoder
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for pixel values in range [0, 1]
#         return x


# class Encoder_LSTM_Conv(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Encoder, self).__init__()
#         self.sequence_length = sequence_length
#         self.hidden_size = hidden_size
#         self.feature_size = feature_size
#         self.latent_size = latent_size
#         self.lstm = nn.LSTM(feature_size, self.sequence_length * self.feature_size, num_layers=2, batch_first=True)
#         self.conv1 = nn.Conv1d(in_channels=self.sequence_length * self.feature_size, out_channels=self.hidden_size*2, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=self.hidden_size*2, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool1d(kernel_size=1, stride=2)
#         self.fc = nn.Linear(hidden_size*3, sequence_length * hidden_size)
#         self.fc1 = nn.Linear(self.hidden_size*3, self.hidden_size)
#         self.fc2 = nn.Linear(self.hidden_size, self.latent_size)
#         # self.fc_mu = nn.Linear(hidden_size, latent_size)
#         # self.fc_sigma = nn.Linear(hidden_size, latent_size)

#     def forward(self, x):
#         _, (h_n, _) = self.lstm(x)
#         h_n = h_n[-1]  # Consider only the last layer because the final layer has all the data about the past
#         x = F.relu(self.conv1(x+h_n.unsqueeze(-1)))
#         x = self.pool(x)
#         x = F.relu(self.conv2(x))
#         x = self.pool(x)
#         x = F.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)  # Flatten the output from conv layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x
    
# class Decoder_LSTM_Conv(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(Decoder, self).__init__()
#         self.sequence_length = sequence_length
#         self.hidden_size = hidden_size
#         self.latent_size = latent_size
#         self.feature_size = feature_size
#         # self.lstm = nn.LSTM(hidden_size, feature_size, num_layers=2, batch_first=True)
#         # self.fc1 = nn.Linear(self.latent_size, self.hidden_size)
#         # self.fc = nn.Linear(self.hidden_size, self.hidden_size * self.sequence_length)
#         # self.fc2 = nn.Linear(self.hidden_size, self.hidden_size*3)
#         # self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*3, kernel_size=3, stride=1, padding=1, output_padding=0)
#         # self.conv2 = nn.ConvTranspose1d(in_channels=self.hidden_size*3, out_channels=self.hidden_size*2, kernel_size=3, stride=1, padding=1, output_padding=0)
#         # self.conv3 = nn.ConvTranspose1d(in_channels=self.hidden_size*2, out_channels=self.sequence_length, kernel_size=3, stride=1, padding=1, output_padding=0)
#         # self.fc_sigma = nn.Linear(hidden_size, latent_size)
#         self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
#         self.fc = nn.Linear(latent_size, sequence_length * hidden_size)
#         self.fc1 = nn.Linear(sequence_length * hidden_size, sequence_length) 

#     def forward(self, z):

#         # z = F.relu(self.fc1(z))
#         # z = F.relu(self.fc2(z))
#         # # z = z.view(-1, 128, 7, 7)  # Reshape to match the dimensions before flattening in encoder
#         # z = z.view(z.size(0), self.hidden_size*3, self.feature_size)  # Reshape to match the dimensions before flattening in encoder
#         # z = F.relu(self.conv1(z))
#         # z = F.relu(self.conv2(z))
#         # z = F.relu(self.conv3(z))
#         # z = self.fc(z.squeeze())
#         # z = z.view(-1, self.sequence_length, self.hidden_size) 
#         # # z = z.squeeze(-1)
#         # output, _ = self.lstm(z)
#         # return torch.sigmoid(output)

#         z = self.fc(z)
#         z = z.view(-1, self.sequence_length, self.hidden_size)  # Add a dimension for sequence length
#         output, _ = self.lstm(z)
#         output = output.reshape(output.size(0), -1)
#         return self.fc1(output)




# class Discriminator(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size):
#         super(Discriminator, self).__init__()
#         self.sequence_length = sequence_length
#         self.feature_size = feature_size
#         self.hidden_size = hidden_size
        
#         self.conv1 = nn.Conv1d(in_channels=self.sequence_length * self.feature_size, out_channels=self.hidden_size, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size*2, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv1d(in_channels=self.hidden_size*2, out_channels=1, kernel_size=3, stride=1, padding=1)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         x = torch.sigmoid(self.conv3(x))  # Sigmoid activation for binary classification
#         return x.squeeze()


# class GAN(nn.Module):
#     def __init__(self, sequence_length, hidden_size, feature_size, latent_size):
#         super(GAN, self).__init__()
#         self.generator = Generator(sequence_length, hidden_size, feature_size, latent_size)
#         self.discriminator = Discriminator(sequence_length, feature_size)

#     def forward(self, z):
#         return self.generator(z)
