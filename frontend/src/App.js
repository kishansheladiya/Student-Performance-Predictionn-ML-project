import React from 'react';
import { Container, Typography, Paper, Box } from '@mui/material';
import PredictionForm from './components/PredictionForm';

function App() {
  return (
    <Container maxWidth="md">
      <Box my={4}>
        <Typography variant="h4" component="h1" gutterBottom>
          Student Performance Prediction
        </Typography>
        <Paper elevation={3} style={{ padding: '1rem' }}>
          <PredictionForm />
        </Paper>
      </Box>
    </Container>
  );
}

export default App;
