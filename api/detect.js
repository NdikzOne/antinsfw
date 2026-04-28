const { IncomingForm } = require('formidable');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

// Matikan koneksi database opsional
export const config = {
  api: {
    bodyParser: false,
  },
};

let model = null;
let modelLoading = false;

// Class names untuk NSFW
const CLASS_NAMES = ['Drawing', 'Hentai', 'Neutral', 'Porn', 'Sexy'];

async function loadModel() {
  if (model) return model;
  if (modelLoading) {
    while (modelLoading) await new Promise(r => setTimeout(r, 100));
    return model;
  }

  modelLoading = true;
  try {
    const modelPath = path.join(process.cwd(), 'model', 'model.json');
    console.log('Loading model from:', modelPath);
    
    model = await tf.loadLayersModel(`file://${modelPath}`);
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Failed to load model:', error);
    throw error;
  } finally {
    modelLoading = false;
  }
  return model;
}

async function preprocessImage(buffer) {
  // Resize ke 224x224 seperti yang diharapkan model
  const image = await sharp(buffer)
    .resize(224, 224, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer();
  
  // Konversi ke tensor dan normalisasi
  let tensor = tf.tensor3d(new Uint8Array(image), [224, 224, 3]);
  tensor = tensor.toFloat();
  tensor = tensor.div(255.0); // Normalize ke [0,1]
  tensor = tensor.expandDims(0); // Tambah batch dimension
  
  return tensor;
}

// Fungsi untuk parsing multipart/form-data dengan formidable
async function parseForm(req) {
  const form = new IncomingForm({
    keepExtensions: true,
    maxFileSize: 10 * 1024 * 1024, // 10MB
  });
  
  return new Promise((resolve, reject) => {
    form.parse(req, (err, fields, files) => {
      if (err) reject(err);
      resolve({ fields, files });
    });
  });
}

module.exports = async (req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed. Use POST.' });
  }
  
  try {
    // Load model
    const nsfwModel = await loadModel();
    
    // Parse form data
    const { files } = await parseForm(req);
    
    // Ambil file yang diupload
    let imageFile = files.image || files.file;
    if (!imageFile) {
      return res.status(400).json({ error: 'No image file provided. Use field name "image" or "file"' });
    }
    
    // Jika array, ambil yang pertama
    if (Array.isArray(imageFile)) imageFile = imageFile[0];
    
    // Baca file buffer
    const buffer = fs.readFileSync(imageFile.filepath);
    
    // Hapus file temporary
    fs.unlinkSync(imageFile.filepath);
    
    // Preprocess image
    const tensor = await preprocessImage(buffer);
    
    // Run inference
    const predictions = await nsfwModel.predict(tensor).data();
    tensor.dispose();
    
    // Format hasil
    const results = CLASS_NAMES.map((name, index) => ({
      className: name,
      probability: parseFloat(predictions[index].toFixed(4))
    }));
    
    // Sort by probability descending
    results.sort((a, b) => b.probability - a.probability);
    
    // Hitung NSFW score (Hentai + Porn + Sexy)
    const nsfwScore = results
      .filter(r => ['Hentai', 'Porn', 'Sexy'].includes(r.className))
      .reduce((sum, r) => sum + r.probability, 0);
    
    const isNSFW = nsfwScore > 0.5;
    
    // Response
    return res.status(200).json({
      success: true,
      prediction: isNSFW ? 'NSFW' : 'SAFE',
      confidence: parseFloat((isNSFW ? nsfwScore : 1 - nsfwScore).toFixed(4)),
      nsfwScore: parseFloat(nsfwScore.toFixed(4)),
      details: results,
      timestamp: new Date().toISOString()
    });
    
  } catch (error) {
    console.error('Detection error:', error);
    return res.status(500).json({
      success: false,
      error: error.message
    });
  }
};