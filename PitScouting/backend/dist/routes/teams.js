"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const multer_1 = __importDefault(require("multer"));
const path_1 = __importDefault(require("path"));
const team_controller_1 = require("../controllers/team.controller");
const router = express_1.default.Router();
const storage = multer_1.default.diskStorage({
    destination: './uploads/',
    filename: (_req, file, cb) => {
        cb(null, `${Date.now()}${path_1.default.extname(file.originalname)}`);
    }
});
const upload = (0, multer_1.default)({ storage });
router.post('/', upload.single('robotImage'), team_controller_1.createTeam);
router.get('/:teamNumber', team_controller_1.getTeam);
router.get('/', team_controller_1.getAllTeams);
router.put('/:teamNumber', upload.single('robotImage'), team_controller_1.updateTeam);
exports.default = router;
//# sourceMappingURL=teams.js.map