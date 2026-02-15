const VPS_URL = process.env.VPS_URL || 'http://104.128.63.239:8080';

module.exports = async function handler(req, res) {
  const path = req.query.path;
  const vpsPath = Array.isArray(path) ? path.join('/') : (path || '');
  const targetUrl = `${VPS_URL}/api/${vpsPath}`;

  try {
    const fetchOptions = {
      method: req.method,
      headers: { 'Content-Type': 'application/json' },
    };

    if (req.method !== 'GET' && req.method !== 'HEAD' && req.body) {
      fetchOptions.body = JSON.stringify(req.body);
    }

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    fetchOptions.signal = controller.signal;

    const response = await fetch(targetUrl, fetchOptions);
    clearTimeout(timeout);
    const data = await response.json();

    res.status(response.status).json(data);
  } catch (error) {
    res.status(502).json({
      error: 'VPS unreachable',
      message: 'Could not connect to the trading server. It may be offline.',
    });
  }
};
