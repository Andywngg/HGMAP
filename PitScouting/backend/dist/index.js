"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const path_1 = __importDefault(require("path"));
const fs_1 = __importDefault(require("fs"));
const models_1 = require("./models");
const auth_1 = __importDefault(require("./routes/auth"));
const teams_1 = __importDefault(require("./routes/teams"));
const app = (0, express_1.default)();
const port = process.env.PORT || 5001;
const env = process.env.NODE_ENV || 'development';
const corsOrigin = env === 'production'
    ? process.env.FRONTEND_URL || 'https://your-frontend-url.vercel.app'
    : 'http://localhost:3000';
app.use((0, cors_1.default)({
    origin: corsOrigin,
    credentials: true
}));
app.use(express_1.default.json());
const uploadsDir = path_1.default.join(__dirname, '../uploads');
if (!fs_1.default.existsSync(uploadsDir)) {
    fs_1.default.mkdirSync(uploadsDir, { recursive: true });
}
app.use('/storage', express_1.default.static(uploadsDir));
app.use('/api/auth', auth_1.default);
app.use('/api/teams', teams_1.default);
app.get('/', (_req, res) => {
    res.json({
        message: 'Scouting App API is running',
        environment: env,
        corsOrigin
    });
});
models_1.sequelize.authenticate()
    .then(() => {
    console.log('Database connected successfully');
    app.listen(port, () => {
        console.log(`Server running on port ${port}`);
        console.log(`Environment: ${env}`);
        console.log(`CORS origin: ${corsOrigin}`);
        console.log('API endpoints:');
        console.log('- POST /api/auth/login');
        console.log('- POST /api/auth/register');
        console.log('- GET /api/teams');
        console.log('- POST /api/teams');
        console.log('- GET /api/teams/:teamNumber');
    });
})
    .catch((error) => {
    console.error('Unable to connect to the database:', error);
});
exports.default = app;
//# sourceMappingURL=index.js.map