"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Team = void 0;
const sequelize_1 = require("sequelize");
class Team extends sequelize_1.Model {
    static initialize(sequelize) {
        this.init({
            id: {
                type: sequelize_1.DataTypes.INTEGER,
                autoIncrement: true,
                primaryKey: true,
            },
            teamNumber: {
                type: sequelize_1.DataTypes.INTEGER,
                allowNull: false,
                unique: true,
            },
            autoScoreCoral: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            },
            autoScoreAlgae: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            },
            autoStartingPosition: {
                type: sequelize_1.DataTypes.STRING,
            },
            teleopCoralCapability: {
                type: sequelize_1.DataTypes.INTEGER,
                defaultValue: 0,
            },
            teleopAlgaeCapability: {
                type: sequelize_1.DataTypes.INTEGER,
                defaultValue: 0,
            },
            teleopDealgifying: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            },
            teleopCoralLevels: {
                type: sequelize_1.DataTypes.ARRAY(sequelize_1.DataTypes.STRING),
                defaultValue: [],
            },
            endgameDeepPerformance: {
                type: sequelize_1.DataTypes.INTEGER,
                defaultValue: 0,
            },
            endgameShallowPerformance: {
                type: sequelize_1.DataTypes.INTEGER,
                defaultValue: 0,
            },
            endgameClimbing: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            },
            robotWidth: {
                type: sequelize_1.DataTypes.FLOAT,
            },
            robotLength: {
                type: sequelize_1.DataTypes.FLOAT,
            },
            robotHeight: {
                type: sequelize_1.DataTypes.FLOAT,
            },
            robotWeight: {
                type: sequelize_1.DataTypes.FLOAT,
            },
            drivetrainType: {
                type: sequelize_1.DataTypes.STRING,
            },
            notes: {
                type: sequelize_1.DataTypes.TEXT,
            },
            imageUrl: {
                type: sequelize_1.DataTypes.STRING,
            },
        }, {
            sequelize,
            tableName: 'teams',
        });
    }
}
exports.Team = Team;
//# sourceMappingURL=team.js.map