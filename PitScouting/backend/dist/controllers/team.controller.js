"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.getAllTeams = exports.updateTeam = exports.getTeam = exports.createTeam = void 0;
const models_1 = require("../models");
const sequelize_1 = require("sequelize");
const createTeam = async (req, res) => {
    try {
        const team = await models_1.Team.create(req.body);
        res.status(201).json(team);
    }
    catch (error) {
        res.status(400).json({ error: 'Error creating team' });
    }
};
exports.createTeam = createTeam;
const getTeam = async (req, res) => {
    try {
        const team = await models_1.Team.findOne({
            where: { teamNumber: req.params.teamNumber },
        });
        if (!team) {
            return res.status(404).json({ error: 'Team not found' });
        }
        return res.json(team);
    }
    catch (error) {
        return res.status(400).json({ error: 'Error fetching team' });
    }
};
exports.getTeam = getTeam;
const updateTeam = async (req, res) => {
    try {
        const team = await models_1.Team.findOne({
            where: { teamNumber: req.params.teamNumber },
        });
        if (!team) {
            return res.status(404).json({ error: 'Team not found' });
        }
        await team.update(req.body);
        return res.json(team);
    }
    catch (error) {
        return res.status(400).json({ error: 'Error updating team' });
    }
};
exports.updateTeam = updateTeam;
const getAllTeams = async (req, res) => {
    try {
        const { search, drivetrain } = req.query;
        const where = {};
        if (search) {
            where.teamNumber = { [sequelize_1.Op.like]: `%${search}%` };
        }
        if (drivetrain) {
            where.drivetrainType = drivetrain;
        }
        const teams = await models_1.Team.findAll({ where });
        res.json(teams);
    }
    catch (error) {
        res.status(400).json({ error: 'Error fetching teams' });
    }
};
exports.getAllTeams = getAllTeams;
//# sourceMappingURL=team.controller.js.map