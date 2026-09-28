# MindTrack frontend

The Colab edition uses a responsive HTML/CSS/JavaScript dashboard served by
FastAPI. It needs no Node installation or frontend build.

- `index.html`: assessment, results, and research views.
- `src/styles/main.css`: responsive layout and visual styling.
- `src/services/app.js`: form generation, API requests, results, and charts.

Run it using the main Colab notebook's API and dashboard cells. The dashboard
calls the actual `/predict`, `/model-info`, and `/metrics` routes on the same
Colab origin. Do not open the HTML file directly from your local computer.
