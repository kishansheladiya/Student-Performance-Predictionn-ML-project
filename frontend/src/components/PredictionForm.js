import React, { useState, useEffect } from 'react';
import { TextField, Button, Grid, Typography, List, ListItem, ListItemText } from '@mui/material';
import axios from 'axios';

const DEFAULT_FEATURES = [
  'sex', 'age', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery',
  'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences', 'G1', 'G2'
]

export default function PredictionForm() {
  const [features, setFeatures] = useState({});
  const [modelInfo, setModelInfo] = useState(null);
  const [result, setResult] = useState(null);

  useEffect(() => {
    // init default values
    const init = {};
    DEFAULT_FEATURES.forEach(f => init[f] = '');
    setFeatures(init);

    // fetch model info
    axios.get('/api/model-info').then(res => setModelInfo(res.data)).catch(() => {});
  }, []);

  const handleChange = (k, v) => {
    setFeatures(prev => ({...prev, [k]: v}));
  }

  const predict = async () => {
    // convert numeric fields
    const payload = {};
    Object.keys(features).forEach(k => {
      const val = features[k];
      if (val === '') return payload[k] = null;
      // try numeric
      const n = Number(val);
      payload[k] = isNaN(n) ? val : n;
    })
    try {
      const res = await axios.post('/api/predict', {features: payload});
      setResult(res.data);
    } catch (err) {
      alert('Prediction failed: ' + (err.response?.data?.detail || err.message));
    }
  }

  return (
    <div>
      <Typography variant="h6">Input student features</Typography>
      <Grid container spacing={2}>
        {DEFAULT_FEATURES.map(f => (
          <Grid item xs={12} sm={4} key={f}>
            <TextField fullWidth label={f} value={features[f] || ''} onChange={e => handleChange(f, e.target.value)} />
          </Grid>
        ))}
      </Grid>
      <Button variant="contained" color="primary" style={{marginTop: '1rem'}} onClick={predict}>Predict</Button>

      {result && (
        <div style={{marginTop: '1rem'}}>
          <Typography variant="h6">Prediction</Typography>
          <Typography>Predicted final score (G3): <strong>{result.prediction.toFixed(2)}</strong></Typography>
          <Typography>Uncertainty: {result.uncertainty !== null ? result.uncertainty.toFixed(3) : 'N/A'}</Typography>

          <Typography variant="h6" style={{marginTop: '0.5rem'}}>Top features</Typography>
          <List>
            {result.top_features.map((t, i) => (
              <ListItem key={i}><ListItemText primary={`${t.feature}: ${t.value} — contribution ${t.contribution}`} /></ListItem>
            ))}
          </List>
        </div>
      )}

      {modelInfo && (
        <div style={{marginTop: '1rem'}}>
          <Typography variant="subtitle1">Model: {modelInfo.model_name} ({modelInfo.model_version})</Typography>
          <Typography variant="body2">Algorithm: {modelInfo.best_algo}</Typography>
          <Typography variant="body2">Metrics: RMSE {modelInfo.metrics.rmse.toFixed(3)}, R2 {modelInfo.metrics.r2.toFixed(3)}</Typography>
        </div>
      )}
    </div>
  )
}
