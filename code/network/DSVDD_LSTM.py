import torch.nn as nn
import torch
# import easydict
# from torchinfo import summary


class Deep_SVDD_LSTM(nn.Module):
    def __init__(self, args):
        super(Deep_SVDD_LSTM, self).__init__()
        self.input_size = args.image_width
        self.hidden_size = args.latent_dim

        self.lstm = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)

    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class pretrain_autoencoder_LSTM(nn.Module):
    def __init__(self, args):
        super(pretrain_autoencoder_LSTM, self).__init__()
        self.input_size = args.image_width
        self.seq_length = args.image_height
        self.hidden_size = args.latent_dim
        self.batch_size = args.batch_size

        self.lstm_en = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)
        self.lstm_de = nn.LSTM(self.input_size, self.hidden_size, 2, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.input_size, bias=False)

    def encoder(self, x):
        outputs, (hidden, cell) = self.lstm_en(x)
        return (hidden, cell)

    def decoder(self, x, hidden_tuple):
        output, (hidden, cell) = self.lstm_de(x, hidden_tuple)
        prediction = self.linear(output)
        return prediction, (hidden, cell)

    def forward(self, x):
        reconstruct_output = []
        x = x.reshape((self.batch_size, self.seq_length, self.input_size))
        hidden = self.encoder(x)
        temp_input = torch.zeros((self.batch_size, 1, self.input_size), dtype=torch.float).to(device)
        for t in range(self.seq_length):
            temp_input, hidden = self.decoder(temp_input, hidden)
            reconstruct_output.append(temp_input)
        reconstruct_output = torch.cat(reconstruct_output, dim=1)
        return reconstruct_output


        # print(x.shape)
        # (batch, seq_length, input_size) --> (batch_size, seq_length, hidden_size), ((num_layers, batch_size, hidden_size), ((num_layers, batch_size, hidden_size)))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# args = easydict.EasyDict({'image_width': 406,
#                           'image_height': 1000,
#                           'latent_dim': 128,
#                           'batch_size': 1
#                           })
# ae = pretrain_autoencoder_LSTM(args).to(device)
# summary(ae, (args.batch_size, 1000, 406), row_settings=["var_names"],
#         col_names=["input_size", "output_size", "num_params"])


# def reconstruct(self, src):
#     batch_size, seq_length, img_size = src.size()

#     # Encoder 넣기
#     hidden = self.encoder(src)

#     outputs = []
#     temp_input = torch.zeros((batch_size, 1, img_size), dtype=torch.float).to(src.device)
#     for t in range(seq_length):
#         temp_input, hidden = self.reconstruct_decoder(temp_input, hidden)
#         outputs.append(temp_input)

#     return torch.cat(outputs, dim=1)
