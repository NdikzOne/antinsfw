import * as tf from '@tensorflow/tfjs-node';
import nsfwjs from 'nsfwjs';
import formidable from 'formidable';
import { readFile, unlink } from 'fs/promises';

let model = null;

async function loadModel() {
  if (!model) {
    console.log('Loading NSFW model...');
    model = await nsfwjs.load('https://cdn.jsdelivr.net/gh/infinitered/nsfwjs@master/models/mobilenet_v2/');
    console.log('Model loaded!');
  }
  return model;
}

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }
  
  if (req.method === 'GET') {
    return res.status(200).json({ 
      status: 'online', 
      message: 'NSFW Detector API is running',
      endpoint: 'POST /api/detect'
    });
  }
  
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  try {
    const form = formidable({
      maxFileSize: 10 * 1024 * 1024,
      allowEmptyFiles: false,
    });
    
    const [fields, files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });
    
    let imageBuffer;
    
    // Cek apakah ada file upload
    if (files.image && files.image[0]) {
      const filePath = files.image[0].filepath;
      imageBuffer = await readFile(filePath);
      await unlink(filePath).catch(() => {});
    }
    // Cek apakah ada URL
    else if (fields.url && fields.url[0]) {
      const response = await fetch(fields.url[0]);
      imageBuffer = Buffer.from(await response.arrayBuffer());
    }
    else {
      return res.status(400).json({ error: 'No image provided' });
    }
    
    // Load model & detect
    const nsfwModel = await loadModel();
    const imageTensor = tf.node.decodeImage(imageBuffer, 3);
    const resized = tf.image.resizeBilinear(imageTensor, [224, 224]);
    const predictions = await nsfwModel.classify(resized);
    
    // Cleanup
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
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
}