/**
 * CDR E2E Test Server
 * 
 * Serves static UI files and proxies /api/v1/* to backend
 * 
 * Usage: node e2e-server.js
 */

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

const UI_PORT = 5173;
const API_PORT = 8000;
const API_HOST = 'localhost';

const MIME_TYPES = {
  '.html': 'text/html',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

const DIST_DIR = path.join(__dirname, 'dist');

function serveStatic(req, res) {
  let filePath = path.join(DIST_DIR, req.url === '/' ? 'index.html' : req.url);
  
  // SPA fallback: if file doesn't exist and not an asset, serve index.html
  if (!fs.existsSync(filePath)) {
    const ext = path.extname(filePath);
    if (!ext || ext === '.html') {
      filePath = path.join(DIST_DIR, 'index.html');
    }
  }
  
  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';
  
  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not Found');
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
}

function proxyToApi(req, res) {
  const options = {
    hostname: API_HOST,
    port: API_PORT,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: `${API_HOST}:${API_PORT}` },
  };
  
  const proxyReq = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res);
  });
  
  proxyReq.on('error', (err) => {
    console.error('Proxy error:', err.message);
    res.writeHead(502);
    res.end(JSON.stringify({ error: 'Backend unavailable', detail: err.message }));
  });
  
  req.pipe(proxyReq);
}

const server = http.createServer((req, res) => {
  // CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  
  if (req.method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }
  
  // Proxy API requests to backend
  if (req.url.startsWith('/api/')) {
    proxyToApi(req, res);
  } else {
    serveStatic(req, res);
  }
});

server.listen(UI_PORT, '0.0.0.0', () => {
  console.log(`🚀 CDR E2E Server running on http://localhost:${UI_PORT}`);
  console.log(`   - UI: static files from ./dist`);
  console.log(`   - API: proxying to http://${API_HOST}:${API_PORT}`);
});
