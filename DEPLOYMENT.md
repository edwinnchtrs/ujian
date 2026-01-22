# 🚀 Deployment Guide - GitHub to Hugging Face Spaces

## 📋 Setup Auto-Deployment

Aplikasi ini sudah dikonfigurasi untuk **automatic deployment** dari GitHub ke Hugging Face Spaces menggunakan GitHub Actions.

---

## 🔐 Langkah 1: Setup GitHub Secret

**Anda sudah memiliki token Hugging Face** (lihat di: https://huggingface.co/settings/tokens)

### Cara Menambahkan Secret ke GitHub:

1. **Buka Repository GitHub Anda**
   - Navigate ke: `https://github.com/YOUR_USERNAME/YOUR_REPO`

2. **Masuk ke Settings**
   - Klik tab **Settings** di repository

3. **Buka Secrets and Variables**
   - Di sidebar kiri, klik **Secrets and variables**
   - Pilih **Actions**

4. **Tambahkan Secret Baru**
   - Klik tombol **New repository secret**
   - **Name**: `HF_TOKEN`
   - **Value**: `<YOUR_HUGGING_FACE_TOKEN>` (dari https://huggingface.co/settings/tokens)
   - Klik **Add secret**

> ⚠️ **PENTING**: Jangan pernah commit token langsung ke code!

---

## ⚙️ Langkah 2: Cara Kerja Auto-Deployment

Workflow GitHub Actions sudah dikonfigurasi di `.github/workflows/deploy-to-huggingface.yml`

### Trigger Deployment:

**Otomatis** setiap kali Anda push ke branch `main` atau `master`:
```bash
git add .
git commit -m "Update aplikasi"
git push origin main
```

**Manual** melalui GitHub UI:
1. Buka tab **Actions** di repository
2. Pilih workflow **Deploy to Hugging Face Spaces**
3. Klik **Run workflow**

---

## 📊 Monitoring Deployment

### Cek Status di GitHub:
1. Buka tab **Actions** di repository
2. Lihat status workflow (✅ success atau ❌ failed)
3. Klik workflow untuk melihat logs detail

### Cek Status di Hugging Face:
1. Buka: `https://huggingface.co/spaces/edwinnchtrs/ujian`
2. Tab **Settings** > **View build logs**
3. Monitor status deployment dan container logs

---

## 🔄 Workflow Process

```mermaid
graph LR
    A[Push to GitHub] --> B[GitHub Actions Trigger]
    B --> C[Checkout Code]
    C --> D[Configure Git]
    D --> E[Push to Hugging Face]
    E --> F[HF Rebuilds Docker]
    F --> G[App Live!]
```

---

## 🛠️ Troubleshooting

### ❌ Deployment Failed

**Problem**: Invalid HF_TOKEN
```
Solution: Pastikan token di GitHub Secrets benar dan masih valid
```

**Problem**: Permission denied
```
Solution: Pastikan token memiliki write access ke Space
```

**Problem**: Branch mismatch
```
Solution: Pastikan branch yang di-push adalah 'main'
```

### 🔍 Debugging

Lihat logs di GitHub Actions:
```bash
# Clone dan test locally
git clone https://github.com/YOUR_USERNAME/YOUR_REPO
cd YOUR_REPO
docker build -t fuel-ai .
docker run -p 7860:7860 fuel-ai
```

---

## 🎯 Hugging Face Space Details

- **Space Name**: `edwinnchtrs/ujian`
- **SDK**: Docker
- **Port**: 7860
- **URL**: `https://huggingface.co/spaces/edwinnchtrs/ujian`

---

## 📝 Manual Deployment (Alternative)

Jika ingin deploy manual tanpa GitHub Actions:

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO
cd YOUR_REPO

# 2. Add Hugging Face remote
git remote add huggingface https://edwinnchtrs:<YOUR_HF_TOKEN>@huggingface.co/spaces/edwinnchtrs/ujian

# 3. Push to Hugging Face
git push huggingface main
```

⚠️ **Security Note**: Ganti token dengan environment variable untuk keamanan:
```bash
export HF_TOKEN="<YOUR_HUGGING_FACE_TOKEN>"
git remote add huggingface https://edwinnchtrs:$HF_TOKEN@huggingface.co/spaces/edwinnchtrs/ujian
```

---

## ✅ Verifikasi Deployment

Setelah deployment berhasil:

1. **Cek Build Status**
   ```
   https://huggingface.co/spaces/edwinnchtrs/ujian
   ```

2. **Test Aplikasi**
   - Buka URL Space
   - Test semua fitur utama
   - Cek dashboard dan visualizations

3. **Monitor Logs**
   - Tab Settings → View logs
   - Pastikan tidak ada errors

---

## 🎉 Next Steps

1. ✅ Setup GitHub Secret (HF_TOKEN)
2. ✅ Push changes ke GitHub
3. ✅ Watch GitHub Actions run
4. ✅ Verify deployment di Hugging Face
5. 🚀 Aplikasi live!

---

## 📚 Resources

- [Hugging Face Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
- [Docker SDK Guide](https://huggingface.co/docs/hub/spaces-sdks-docker)

---

**Made with ❤️ - Auto-deployment powered by GitHub Actions**
