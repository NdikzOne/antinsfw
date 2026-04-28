import * as tf from '@tensorflow/tfjs';
import nsfwjs from 'nsfwjs';
import { readFile, unlink } from 'fs/promises';
import formidable from 'formidable';

// JANGAN simpan model di function, load dari CDN setiap request
// Atau cache di memory global (masih dalam batas)

let modelPromise = null;

async function getModel() {
  if (!modelPromise) {
    console.log('Loading NSFW model from CDN...');
    // Load dari CDN - tidak perlu model di bundle
    modelPromise = nsfwjs.load('https://cdn.jsdelivr.net/gh/infinitered/nsfwjs@master/models/');
  }
  return modelPromise;
}

export const config = {
  api: {
    bodyParser: false,
    maxDuration: 10, // Maksimal 10 detik
  },
};

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  
  if (req.method === 'OPTIONS') return res.status(200).end();
  
  if (req.method === 'GET') {
    return res.status(200).json({ 
      status: 'online',
      note: 'NSFW Detector API (lightweight mode)'
    });
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const form = formidable({ maxFileSize: 5 * 1024 * 1024 });
    const [fields, files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });
    
    let imageBuffer;
    
    if (files.image && files.image[0]) {
      const filePath = files.image[0].filepath;
      imageBuffer = await readFile(filePath);
      await unlink(filePath).catch(() => {});
    } else if (fields.url && fields.url[0]) {
      const response = await fetch(fields.url[0]);
      imageBuffer = Buffer.from(await response.arrayBuffer());
    } else {
      return res.status(400).json({ error: 'No image provided' });
    }
    
    // Load model (dari CDN, tidak disimpan di function)
    const model = await getModel();
    
    // Decode image
    const imageTensor = tf.node.decodeImage(imageBuffer, 3);
    const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const predictions = await model.classify(resized);
    
    imageTensor.dispose();
    resized.dispose();
    
    const nsfwCategories = ['Hentai', 'Porn', 'Sexy'];
    const nsfwScore = predictions
      .filter(p => nsfwCategories.includes(p.className))
      .reduce((sum, p) => sum + p.probability, 0);
    
    res.status(200).json({
      success: true,
      nsfw: nsfwScore > 0.5,
      nsfwScore: nsfwScore,
      predictions: predictions.map(p => ({
        label: p.className,
        confidence: p.probability,
        percentage: `${(p.probability * 100).toFixed(2)}%`
      }))
    });
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ success: false, error: error.message });
  }
}