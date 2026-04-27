const nsfw = require('nsfwjs');
const tf = require('@tensorflow/tfjs-node');
const sharp = require('sharp');
const multer = require('multer');

// Disable tf logging
tf.enableProdMode();

let model = null;

// Load model on startup
async function loadModel() {
  if (!model) {
    console.log('Loading NSFW model...');
    // Use local model or from CDN
    model = await nsfw.load('file://model/', {
      size: 224,
      type: 'graph'
    });
    console.log('Model loaded successfully');
  }
  return model;
}

// Configure multer for memory storage
const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

// Helper function to process image
async function processImage(buffer) {
  try {
    // Resize image to 224x224 using sharp
    const image = await sharp(buffer)
      .resize(224, 224, { fit: 'cover' })
      .toBuffer();
    
    // Convert to tensor
    const decoded = tf.node.decodeImage(image, 3);
    const tensor = decoded.expandDims(0);
    
    return tensor;
  } catch (error) {
    console.error('Image processing error:', error);
    throw new Error('Failed to process image');
  }
}

export const config = {
  api: {
    bodyParser: false,
    externalResolver: true,
  },
};

export default async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    // Load model
    await loadModel();

    // Handle file upload
    await new Promise((resolve, reject) => {
      upload.single('image')(req, res, (err) => {
        if (err) reject(err);
        else resolve();
      });
    });

    if (!req.file) {
      return res.status(400).json({ error: 'No image provided' });
    }

    // Process image
    const tensor = await processImage(req.file.buffer);
    
    // Classify
    const predictions = await model.classify(tensor);
    
    // Clean up tensor
    tensor.dispose();
    
    // Calculate NSFW score
    const nsfwScore = predictions
      .filter(p => ['Hentai', 'Porn', 'Sexy'].includes(p.className))
      .reduce((sum, p) => sum + p.probability, 0);
    
    const isNSFW = nsfwScore > 0.5;
    
    // Format response
    const response = {
      success: true,
      isNSFW,
      nsfwScore: parseFloat(nsfwScore.toFixed(4)),
      predictions: predictions.map(p => ({
        className: p.className,
        probability: parseFloat(p.probability.toFixed(4))
      })),
      timestamp: new Date().toISOString()
    };
    
    res.status(200).json(response);
    
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ 
      success: false, 
      error: error.message || 'Internal server error' 
    });
  }
}