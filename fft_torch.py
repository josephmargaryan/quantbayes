import torch
import torch.nn as nn

class FFTCirculantLayer(nn.Module):
    def __init__(self, n):
        super(FFTCirculantLayer, self).__init__()
        self.n = n
        self.c = nn.Parameter(
            torch.randn(n)
        )  # Learnable parameters for the circulant matrix

    def forward(self, x):
        # FFT-based multiplication
        c_fft = torch.fft.fft(self.c)
        x_fft = torch.fft.fft(x, n=self.n)
        result_fft = c_fft * x_fft
        result = torch.fft.ifft(result_fft)
        return result.real


class FFTModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FFTModel, self).__init__()
        self.layer1 = FFTCirculantLayer(input_dim)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.fc(x)
        return x
