const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());

// Routes
app.use('/api/auth', require('./routes/auth'));
app.use('/api/profile', require('./routes/profile'));
app.use('/api/foods', require('./routes/foods'));
app.use('/api/recommend', require('./routes/recommendations'));
app.use('/api/logs', require('./routes/logs'));
app.use('/api/habits', require('./routes/habits'));

app.get('/api/health', (req, res) => res.json({ status: 'NutriTrack AI Backend Running ✅' }));

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`🚀 Backend running on http://localhost:${PORT}`));
