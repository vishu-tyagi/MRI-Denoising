import numpy as np
import scipy.stats as stats
from skimage.metrics import structural_similarity


class CustomEvaluation():
    def __init__(self):
        pass

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """
        Obtain normalized absolute array
        Args:
            X (np.ndarray): Input array such as k-space array
        Returns:
            np.ndarray: Normalized absolute array
        """
        Y = np.abs(X)
        Y = np.average(Y, axis=0)
        Y = (Y - np.min(Y)) / (np.max(Y) - np.min(Y))
        return Y

    def kspace_to_mri(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse Fourier transform k-space array to obtain MR image array
        Args:
            X (np.ndarray): k-space array
        Returns:
            np.ndarray: MR image array
        """
        Y = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(X)))
        Y = self.normalize(Y)
        return Y

    def compute_psnr(self, original: np.ndarray, compressed: np.ndarray):
        """
        Calculate Peak Signal to Noise Ratio
        """
        mse = np.mean((original - compressed) ** 2)
        # Zero MSE means no noise is present in the signal
        # Therefore PSNR have no importance
        if not mse:
            return 100
        max_pixel = 1.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def compute_pcc(self, X: np.ndarray, Y: np.ndarray):
        """
        Calculate Pearson Correlation Coefficient
        """
        return np.corrcoef(np.reshape(X, (-1)), np.reshape(Y, (-1)))[0, 1]

    def compute_ssim(self, X: np.ndarray, Y: np.ndarray):
        """
        Calculate Structural Similarity score
        """
        return structural_similarity(X, Y)

    def compute_scc(self, X: np.ndarray, Y: np.ndarray):
        """
        Calculate Spearman Correlation Coefficient
        """
        X = np.reshape(X, (-1))
        Y = np.reshape(Y, (-1))
        corr, _ = stats.spearmanr(X, Y)
        return corr

    def compute_scores(self, output: np.ndarray, ground_truth: np.ndarray):
        batch_size = ground_truth.shape[0]
        psnr, pcc, ssim, scc = [], [], [], []
        for i in range(batch_size):
            output_mri = self.kspace_to_mri(output[i, ...])
            ground_truth_mri = self.kspace_to_mri(ground_truth[i, ...])
            psnr.append(self.compute_psnr(output_mri, ground_truth_mri))
            pcc.append(self.compute_pcc(output_mri, ground_truth_mri))
            ssim.append(self.compute_ssim(output_mri, ground_truth_mri))
            scc.append(self.compute_scc(output_mri, ground_truth_mri))
        psnr = np.mean(psnr)
        pcc = np.mean(pcc)
        ssim = np.mean(ssim)
        scc = np.mean(scc)
        return [psnr, pcc, ssim, scc]
