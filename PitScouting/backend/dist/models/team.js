"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
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
                allowNull: true,
            },
            teleopDealgifying: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            },
            teleopPreference: {
                type: sequelize_1.DataTypes.STRING,
                allowNull: true,
            },
            scoringPreference: {
                type: sequelize_1.DataTypes.STRING,
                allowNull: true,
            },
            coralLevels: {
                type: sequelize_1.DataTypes.ARRAY(sequelize_1.DataTypes.STRING),
                defaultValue: [],
                get() {
                    const rawValue = this.getDataValue('coralLevels');
                    return rawValue ? rawValue : [];
                },
                set(value) {
                    if (typeof value === 'string') {
                        try {
                            this.setDataValue('coralLevels', JSON.parse(value));
                        }
                        catch (_a) {
                            this.setDataValue('coralLevels', []);
                        }
                    }
                    else if (Array.isArray(value)) {
                        this.setDataValue('coralLevels', value);
                    }
                    else {
                        this.setDataValue('coralLevels', []);
                    }
                }
            },
            endgameType: {
                type: sequelize_1.DataTypes.STRING,
                defaultValue: 'none',
            },
            robotWidth: {
                type: sequelize_1.DataTypes.FLOAT,
                allowNull: true,
            },
            robotLength: {
                type: sequelize_1.DataTypes.FLOAT,
                allowNull: true,
            },
            robotHeight: {
                type: sequelize_1.DataTypes.FLOAT,
                allowNull: true,
            },
            robotWeight: {
                type: sequelize_1.DataTypes.FLOAT,
                allowNull: true,
            },
            drivetrainType: {
                type: sequelize_1.DataTypes.STRING,
                allowNull: true,
            },
            notes: {
                type: sequelize_1.DataTypes.TEXT,
                allowNull: true,
            },
            imageUrl: {
                type: sequelize_1.DataTypes.STRING,
                allowNull: true,
            },
            mustStartSpecificPosition: {
                type: sequelize_1.DataTypes.BOOLEAN,
                defaultValue: false,
            }
        }, {
            sequelize,
            modelName: 'Team',
            tableName: 'teams'
        });
    }
}
exports.default = Team;
//# sourceMappingURL=team.js.map