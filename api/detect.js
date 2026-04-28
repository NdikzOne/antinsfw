import * as tf from '@tensorflow/tfjs-node';
import nsfwjs from 'nsfwjs';
import { IncomingForm } from 'formidable';
import fetch from 'node-fetch';
import { createReadStream } from 'fs';
import { unlink } from 'fs/promises';

let model = null;

async function loadModel() {
  if (!model) {
    // Menggunakan model dari CDN
    model = await nsfwjs.load('https://cdn.jsdelivr.net/gh/infinitered/nsfwjs@master/models/mobilenet_v2/', {
      size: 299,
      type: 'graph'
    });
  }
  return model;
}

// Disable body parser untuk handle file upload
export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method === 'GET') {
    return res.status(200).json({
      status: 'online',
      message: 'NSFW Detector API',
      usage: 'POST /api/detect with multipart/form-data (field: image)'
    });
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Parse form data
    const form = new IncomingForm({
      keepExtensions: true,
      maxFileSize: 10 * 1024 * 1024, // 10MB
    });

    const [fields, files] = await new Promise((resolve, reject) => {
      form.parse(req, (err, fields, files) => {
        if (err) reject(err);
        resolve([fields, files]);
      });
    });

    let fileBuffer = null;
    let filePath = null;

    // Handle file dari upload
    if (files.image && files.image[0]) {
      filePath = files.image[0].filepath;
      const fs = await import('fs');
      fileBuffer = await fs.promises.readFile(filePath);
      await unlink(filePath); // Hapus temporary file
    } 
    // Handle URL image
    else if (fields.url && fields.url[0]) {
      const response = await fetch(fields.url[0]);
      fileBuffer = await response.buffer();
    }
    else {
      return res.status(400).json({ error: 'No image provided. Send file (image) or url' });
    }

    // Load model
    const nsfwModel = await loadModel();

    // Decode image ke tensor
    const imageTensor = tf.node.decodeImage(fileBuffer, 3);
    
    // Resize ke ukuran yang diharapkan model (299x299)
    const resized = tf.image.resizeBilinear(imageTensor, [299, 299]);
    
    // Detect
    const predictions = await nsfwModel.classify(resized);
    
    // Cleanup
    imageTensor.dispose();
    resized.dispose();

    const nsfwScore = predictions
      .filter(p => ['Hentai', 'Porn', 'Sexy'].includes(p.className))
      .reduce((sum, p) => sum + p.probability, 0);
    
    const isNSFW = nsfwScore > 0.5;

    res.status(200).json({
      safe: !isNSFW,
      nsfw: isNSFW,
      score: nsfwScore,
      predictions: predictions.map(p => ({
        className: p.className,
        probability: Number(p.probability.toFixed(4)),
        percentage: `${(p.probability * 100).toFixed(2)}%`
      })),
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('Detection error:', error);
    res.status(500).json({
      error: 'Detection failed',
      message: error.message
    });
  }
}