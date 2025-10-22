Diffusion-VAE — Reconstruction TDI (PyTorch)

Ce dépôt contient un pipeline PyTorch pour entraîner un ConvVAE sur des images TDI (Tissue Doppler Imaging) en niveau de gris (1 canal), avec évaluation via MSE et SSIM, et visualisation des reconstructions. Le code est pensé pour Google Colab + Google Drive, mais peut être exécuté localement avec de légères modifications.

✨ Fonctionnalités

Chargement d’images TDI 1-canal, normalisées et redimensionnées.

ConvVAE avec skip-connections côté décodeur.

Pertes : MSE, 1–SSIM, KL (pondérées).

Journalisation des pertes/metrics, sauvegarde modèle et figures.

Reproductibilité (graine fixée).

Splitting train/val (90/10 par défaut).

⚠️ Le script actuel se concentre sur la reconstruction. L’augmentation par modèles de diffusion n’est pas incluse dans ce fichier (ici : VAE seul).

📁 Structure attendue des données
DATATDI/
├── CTRCD/
│   ├── patient001.png
│   ├── ...
└── NO_CTRCD/
    ├── patient101.png
    ├── ...


Les fichiers .png sont détectés automatiquement. Les sous-dossiers CTRCD et NO_CTRCD sont obligatoires (même si seule la reconstruction est utilisée, cela uniformise l’accès aux images).

🧰 Prérequis

Python 3.9+

PyTorch (CUDA si dispo)

torchvision

numpy, pandas, pillow (PIL)

matplotlib

tqdm

pytorch-msssim

(Colab) monté sur Google Drive

Installation rapide (exemple) :

pip install torch torchvision pytorch-msssim tqdm pillow matplotlib pandas

⚙️ Variables à définir (IMPORTANT)

Le script utilise des constantes non définies dans le snippet. Ajoute-les avant de créer le dataset :

# Hyperparamètres / chemins
IMAGE_SIZE   = (160, 384)   # (H, W) final des trames
BATCH_SIZE   = 16
LATENT_DIM   = 256
LR           = 1e-3
EPOCHS       = 50
ALPHA_SSIM   = 1.0          # poids pour (1-SSIM)
BETA_KL      = 1e-4         # poids KL
OUT_DIR      = os.path.join(BASE_DIR, 'outputs_vae')

os.makedirs(OUT_DIR, exist_ok=True)


💡 Tu peux ajuster BATCH_SIZE selon ta VRAM.
💡 IMAGE_SIZE doit être compatible avec 4 couches de Conv2d stride=2 (divisible par 16).

🚀 Utilisation (Colab)

Monte Google Drive (déjà dans le code) :

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
BASE_DIR  = '/content/drive/MyDrive/colab'
DATA_DIR  = os.path.join(BASE_DIR, 'DATATDI')


Définis les variables manquantes (cf. section précédente).

Lance l’entraînement. Les sorties sont dans OUT_DIR :

vae_tdi.pt

train_loss.png, val_metrics.png

reconstructions.png

🧩 Détails techniques
Dataset
class TDIDataset(Dataset):
    def __init__(self, root, size):
        self.files = []
        for sub in ['CTRCD','NO_CTRCD']:
            d = os.path.join(root, sub)
            for f in os.listdir(d):
                if f.lower().endswith('.png'):
                    self.files.append(os.path.join(d,f))
        self.tf = T.Compose([
            T.Grayscale(1),        # 1 canal
            T.Resize(size),        # (H, W)
            T.ToTensor(),          # [0,1]
            T.Normalize([0.5],[0.5])  # [-1,1]
        ])
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert('RGB')
        return self.tf(img), self.files[i]

Modèle (ConvVAE + skips)

Encodeur : 4 blocs Conv2d (1→32→64→128→256), stride=2.

Bottleneck : fc_mu, fc_logvar, reparamétrisation.

Décodeur : 4 blocs ConvTranspose2d (256→128→64→32→1) + skip-connections (+ e3, e2, e1).

Sortie avec Tanh → plage [-1, 1].

Pertes
mse_fn  = nn.MSELoss()
ssim_fn = SSIM(data_range=2.0, channel=1, win_size=7)  # data_range=2 pour [-1,1]
loss = l_mse + ALPHA_SSIM * (1 - ssim) + BETA_KL * kld

🧪 Validation & courbes

val_mses : MSE moyen par epoch (validation).

val_ssims : SSIM moyen par epoch (validation).

Figures :

train_loss.png

val_metrics.png (MSE, SSIM)

💾 Sauvegardes

Le script sauvegarde :

vae_tdi.pt (poids du VAE).

⚠️ Le snippet montre aussi :

joblib.dump(dataset.scaler,   os.path.join(OUT_DIR, 'scaler.joblib'))
joblib.dump(dataset.encoders, os.path.join(OUT_DIR, 'encoders.joblib'))


Ces objets n’existent pas dans ce fichier (pas de variables cliniques ici).
➡️ Supprime ces lignes ou fournis un dataset tabulaire avec scaler/encoders.

🖼️ Inspection qualitative

Le bloc actuel contient une petite erreur (utilise x sans le charger). Remplace par :

vae.eval()
fig, axs = plt.subplots(5,2,figsize=(6,15))
with torch.no_grad():
    # prends un batch de la validation
    for i, (x, _) in enumerate(va_loader):
        x = x.to(DEVICE)
        recon, _, _ = vae(x)
        # on n'affiche que les 5 premières images du batch
        for k in range(min(5, x.size(0))):
            orig = (x[k]*0.5 + 0.5).clamp(0,1)
            rec  = (recon[k]*0.5 + 0.5).clamp(0,1)
            axs[k,0].imshow(orig[0].cpu(), cmap='gray'); axs[k,0].set_title("Original"); axs[k,0].axis('off')
            axs[k,1].imshow(rec[0].cpu(),  cmap='gray'); axs[k,1].set_title("Reconstr."); axs[k,1].axis('off')
        break  # une itération suffit
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"reconstructions.png"))
plt.show()

🔧 Exécution locale (sans Colab)

Remplace la partie Drive par des chemins locaux :

BASE_DIR = '/chemin/vers/ton/projet'
DATA_DIR = os.path.join(BASE_DIR, 'DATATDI')


Assure-toi que CUDA est disponible sinon l’entraînement passera en CPU.

🧯 Dépannage (FAQ)

NameError: IMAGE_SIZE is not defined
➜ Définis toutes les constantes (cf. section Variables à définir).

joblib is not defined
➜ pip install joblib et import joblib, et assure-toi d’avoir des objets scaler/encoders (sinon supprime ces lignes).

Erreur SSIM / tailles
➜ win_size=7 exige des images ≥ 7×7. Ton IMAGE_SIZE est 160×384 → OK.

Artifacts visuels / reconstructions floues
➜ Essaie d’augmenter LATENT_DIM, baisse BETA_KL, ou entraîne plus longtemps.

Mémoire saturée (CUDA OOM)
➜ Réduis BATCH_SIZE, ou l’IMAGE_SIZE.

📊 Feuille de route (TODO)

 Factoriser la config via YAML/argparse.

 Ajouter k-fold CV et moyennes ± écart-type.

 Logger (TensorBoard/W&B).

 Export ONNX / TorchScript.

 Dataset 2D/3D avec lecture DICOM.

 Module Diffusion pour augmentation.

🔒 Données sensibles

Ne pousse aucune donnée patient ni métadonnée identifiante. Conserve uniquement des données anonymisées ou synthétiques.

📜 Licence

Ajoute un fichier LICENSE (ex. MIT, Apache-2.0). Exemple MIT :

MIT License — Copyright (c) 2025 …

📣 Citation

Si tu utilises ce code dans une publication :

@software{DiffusionVAE_TDI_2025,
  author = {Ton Nom},
  title = {Diffusion-VAE for TDI Reconstruction},
  year = {2025},
  url = {https://github.com/<ton-user>/<ton-repo>}
}

🤝 Contributions

Issues et PR bienvenues ! Merci de respecter PEP8 et d’ajouter des tests simples (chargement dataset, passage avant/arrière du modèle).

📞 Contact

Auteur : Ton Nom

Email : ton.email@exemple.com

Institution : …
