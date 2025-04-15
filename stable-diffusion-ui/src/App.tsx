import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [prompt, setPrompt] = useState('');
  const [image, setImage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const generateImage = async () => {
    setLoading(true);
    setError('');
    setImage('');

    try {
      const response = await axios.post(
        `${process.env.REACT_APP_API_URL}/api/generate`,
        {
          prompt: prompt,
          num_steps: 60,
          guidance_scale: 7.0
        },
        {
          headers: {
            Authorization: `Bearer ${process.env.REACT_APP_API_KEY}`
          }
        }
      );

      setImage(`data:image/png;base64,${response.data.image}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <h1>Stable Diffusion 3.5 Generator</h1>
      <input
        type="text"
        placeholder="Enter your prompt..."
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
      />
      <button onClick={generateImage} disabled={loading}>
        {loading ? 'Generating...' : 'Generate Image'}
      </button>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {image && (
        <div>
          <h3>Generated Image</h3>
          <img src={image} alt="Generated" style={{ maxWidth: '100%' }} />
        </div>
      )}
    </div>
  );
}

export default App;
