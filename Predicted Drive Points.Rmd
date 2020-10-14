---
title: "A Predicted Drive Points Model for College Football"
author: "Conor R. McQuiston"
date: "9/30/2020"
output: html_document
---

## Introduction

In football, not all yards are created equal. What the analytics community means by this is that you can gain the same amount of yards in separate situations, but those gains won't be worth the same amount. For example, gaining 3 yards on 1st and 10 on your own 25 isn't very valuable, but gaining 3 yards on 4th and 1 on your opponents 35 is extremely valuable.

This simple idea has driven the creation of [Expected Points Added (EPA)](https://www.espn.com/nfl/story/_/id/8379024/nfl-explaining-expected-points-metric), which is generally considered to be the fundamental statistic in Football Analytics. EPA has opened avenues previously impossible with just box score stats allowing us to get a more complete picture of the effectiveness of a given play and therefore can easily be expanded to offensive and defensive schemes, teams overall, individual players, etc.

This idea can be easily extended to the drive level. Not all touchdown drives are created equal. What does this mean specifically? Imagine two drives, in one drive a team methodically marches down the length of the field and scores an easy touchdown, and in another a team gains only 1 or 2 yards on early downs, but manages to get just enough yards to convert on 3rd down and eventually stumbles their way into the endzone. I am exerting my own judgment here, but I would consider the first drive to be superior.

Both drives have the same result, but a team would expect to score much more often when performing like the first drive than the second drive. In other words, the first drive has a better expected outcome because the process is better. I believe it would be useful for teams to be able to differentiate between these two drives, so I (with massive help from some very smart people) created this model in an attempt to do so.

The short of this model is that it is takes a team-agnostic look at the beginning game-state of the drive, and the process of the drive and predict how many points will be scored, the Predicted Drive Points (PDP).

------------------------------------------------------------------------

## Mathematical Background

So how did I try and go about modeling this? I decided to do a logistic regression on every scoring outcome a drive can have (Touchdown, Field Goal, Opponent Touchdown, and Safety) in order to get a probability for each in the drive, and use those probabilities to compute an expected value. Expressed mathematically:

PDP = P(TD)\*7 + P(FG)\*3 + P(Opp TD)\*-7 + P(Safety)\*-2

Why did I choose this particular method? Since drives can only score a discrete set of points, a linear regression seemed like a poor choice. I considered using a multinomial logistic regression, however I am not familiar enough with testing processes involving it to feel comfortable using it. Moving forward in making improvements with the model it seems to be the most natural choice if creating an expected value is the best way to achieve this goal.

There are of course issues with this approach. Notably, under the current model \~15% of drives are predicted to score more than 7 points, with a maximum value of 10 predicted points and the probabilities of each type of score can add to more than 1. This is a result of using a series of logistic regressions instead of a multinomial logistic regression. On particularly good drives, the probability that a team will score either a Touchdown *or* Field Goal is high, so the model spits out a high value. Most of these drives result in touchdowns regardless, so it is not too big of a concern, however it is certainly a flaw.

------------------------------------------------------------------------

## **Data Acquisition**

The data was acquired using the cfbscrapR package. I also used several other packages to assist with the data wrangling and evaluating the logistic regressions. Uncomment the first two lines if you need to install cfbscrapR.

```{r}
#install.packages("devtools")
#devtools::install_github("meysubb/cfbscrapR")

library(tidyverse)
library(cfbscrapR)
library(pROC)
```

The first step is to download the data. I built this off of play-by-play data from the College Football Playoff era as it is the most consistent data, and with over 5 years of data it should suffice. This chunk will take a moment as it is \~700 MB of data. We'll also set a seed here to get reproducible results

```{r}
set.seed(472)
all_years <- data.frame()

for(i in 2014:2019){
  for(j in 1:15){
    x <- cfb_pbp_data(year = i, season_type = "both", week = j, epa_wpa = TRUE) %>% mutate(year = i, week = j)
    all_years <- rbind(all_years, x)
  }
}
```

From here I'll begin to wrangle the data into the form I want. This is a good point to list what the inputs are, and why I am including them:

-   **Starting Yardline, Starting Quarter, Starting Time Remaining Until Half, Starting Score Differential -** These are all the same category, they just capture the game state at the beginning of the drive

-   **Classic Success Rate -** % of time the offense gains 40% of yards to go on 1st down, 60% of yards to go on 2nd down, or converts on 3rd or 4th down. This is used to measure the efficiency of the offense.

-   **Points Created -** Total EPA \* Classic Success Rate. This is meant to capture the total production of the offense on the drive. This stat is the brainchild of Parker Fleming (\@statsowar on Twitter)

-   **EPA/Play -** The average amount of EPA the offense gains on a play in the drive. This, unlike Classic Success Rate, also captures *how* successful or unsuccessful a given play was.

-   **EPA Standard Deviation -** This is meant to be a proxy for how consistent an offense was on a given drive

-   **Number of Plays -** Long drives tend to score more often

You will notice that there are some variables in our collection that are not included in the model inputs. Some of these are used for comparison purposes, others are used as target variables for the Logistic Regressions. These should be obvious.

There is a quick intermediate step we have to take. The definition of **Classic Success Rate** that I am using is not represented in the cfbscrapR play by play data. So we have to quickly add that data in:

```{r}
all_years <- all_years %>% mutate(
  succ = ifelse(down == 1, ifelse(yards_gained >= 0.4*distance, 1, 0),
                ifelse(down == 2, ifelse(yards_gained >= 0.6*distance, 1, 0), 
                       ifelse(yards_gained >= distance, 1, 0))
  ))
```

There is also a quick stage of cleaning that we have to do, and that is removing Kickoffs. I have no interest in predicting Kickoff Return TDs, and it has little to no influence on how good a given drive is. So we will make a vector of the Kickoff play types, and then remove those play types from the whole play by play set. Also timeouts, since we don't want timeouts artificially inflating how long a given drive was.

```{r}
kickoffs <- c('Kickoff Return (Offense)', 'Timeout', 'Kickoff', 'Kickoff Return Touchdown')
```

With this out of the way, now we can finally wrangle the data to get it into a form we want:

```{r}
all_drives <- all_years %>% 
  filter(!(play_type %in% kickoffs)) %>% 
  group_by(game_id, drive_number) %>% 
  summarise(
    off = offense_play,
    def = defense_play,
    csr = mean(succ, na.rm = T),
    tepa = sum(EPA, na.rm = T),
    pc = tepa*csr,
    start = drive_start_yards_to_goal,
    score_diff = score_diff_start,
    start_quarter = first(period),
    mean_epa = mean(EPA, na.rm = T),
    num_plays = n(),
    year = year,
    std_epa = ifelse(num_plays != 1, sd(EPA, na.rm = T), 0),
    start_time_till_half_end = drive_time_minutes_start*60*ifelse((start_quarter == 1), 2, 1)*ifelse((start_quarter == 3), 2, 1) + drive_time_seconds_start,
    drive_points = ifelse(drive_result == 'PUNT TD', -1*drive_pts, drive_pts), 
    touchdown = ifelse(drive_result == 'TD', 1,
                                    ifelse(drive_result == 'END OF HALF TD', 1,
                                                  ifelse(drive_result == 'FG TD', 1,
                                                         ifelse(drive_result == 'PUNT TD', 1,
                                                          ifelse(drive_result == 'END OF GAME TD', 1, 0))))),
    fg = ifelse(drive_result == 'FG', 1, 0),
    opp_td = ifelse(drive_result == 'FUMBLE RETURN TD', 1,
                    ifelse(drive_result == 'INT TD', 1,
                           ifelse(drive_result == 'FUMBLE TD', 1,
                           ifelse(drive_result == 'DOWNS TD', 1,
                                ifelse(drive_result == 'PUNT RETURN TD', 1,
                                  ifelse(drive_result == 'MISSED FG TD', 1, 0)))))),
    saf = ifelse(drive_result == 'SF', 1, 0),
    result = drive_result,
    initial_ep = first(ep_before)
  ) %>% 
  distinct() %>% 
  filter(start != 0)
```

Now that we have the data in the form we want, we can finally begin actually building the models.

## Model Building and Testing

The first step in model building is to make our training and testing sets. I'll do this by splitting the entire data frame into 80% training and 20% testing. Why 80/20? Well I used to do 50/50 when I was primarily in Python, but then when I googled how to split data into training and testing sets in R, the code on StackOverflow had an 80/20 split. So here we are.

```{r}
ind <- sample(2, nrow(all_drives), replace = TRUE, prob = c(0.8, 0.2))
drive_train <- all_drives[ind == 1,]
drive_test <- all_drives[ind == 2,]
```

Now we actually build the models, which is only a single line for each one.

```{r}
drive_td_pred <- glm(formula = touchdown ~  mean_epa + std_epa + csr + pc + start + score_diff + num_plays + start_time_till_half_end, data = drive_train, family = binomial, na.action = na.exclude)

drive_fg_pred <- glm(formula = fg ~  mean_epa + std_epa + csr + pc + start + score_diff + num_plays + start_time_till_half_end, data = drive_train, family = binomial)

drive_opp_td_pred <- glm(formula = opp_td ~  mean_epa + std_epa + csr + pc + start + score_diff + num_plays + start_time_till_half_end, data = drive_train, family = binomial)

drive_saf_pred <- glm(formula = saf ~  mean_epa + std_epa + csr + pc + start + score_diff + num_plays + start_time_till_half_end, data = drive_train, family = binomial)
```

There will be a warning that some of the fitted probabilities are 0 or 1. This may be a sign of overfitting the data, but considering the 80/20 split I find that unlikely, and it is only a handful of drives which have a 0 or 1 probability so I don't think it is too concerning.

Now we need to formally test these models. The best way I know to do that for simple logistic regressions is with ROC curves. To make them look nice, we'll have to load in some packages first. So let's look at them one at a time:

```{r}
library(ggplot2)
library(extrafont)
library(ggrepel)
loadfonts(device = "win")

td_roc <- roc(drive_test$touchdown, predict(drive_td_pred, newdata = drive_test))
td_auc <- toString(td_roc$auc)
fg_roc <- roc(drive_test$fg, predict(drive_fg_pred, newdata = drive_test))
fg_auc <- toString(fg_roc$auc)
saf_roc <- roc(drive_test$saf, predict(drive_saf_pred, newdata = drive_test))
saf_auc <- toString(saf_roc$auc)
opp_td_roc <- roc(drive_test$opp_td, predict(drive_opp_td_pred, newdata = drive_test))
opp_td_auc <- toString(opp_td_roc$auc)
```

```{r}
ggtd_roc <- ggroc(td_roc)

ggtd_roc+
  geom_text(mapping = aes(x = 0.5, y = 0.5, label = paste0('AUC of ', td_auc)))+
  labs(title = "TD Logistic Model ROC Curve",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

```{r}
ggfg_roc <- ggroc(fg_roc)

ggfg_roc+
  geom_text(mapping = aes(x = 0.5, y = 0.5, label = paste0('AUC of ', fg_auc)))+
  labs(title = "FG Logistic Model ROC Curve",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

```{r}
ggsaf_roc <- ggroc(saf_roc)

ggsaf_roc+
  geom_text(mapping = aes(x = 0.5, y = 0.5, label = paste0('AUC of ', saf_auc)))+
  labs(title = "Safety Logistic Model ROC Curve",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

```{r}
ggopp_td_roc <- ggroc(opp_td_roc)

ggopp_td_roc+
  geom_text(mapping = aes(x = 0.5, y = 0.5, label = paste0('AUC of ', opp_td_auc)))+
  labs(title = "Opponent TD Logistic Model ROC Curve",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

These values are all really good! The only ones which are less than 0.9 AUC are opponent TDs and field goals. The former is likely because Pick 6s/Scoop n Scores can happen on successful drives and be extremely hard to predict. The latter is likely because very successful drives which can end in touchdowns are also likely to end in field goals. So under the conditions of the current model, I feel fine about this.

This is extremely promising, but now we have to complete the last step in actually calculating the predicted points. And to evaluate whether or not it is acceptable, we'll look at the residual between the predicted and actual drive points.

```{r}
drive_test$pred_td <- predict(drive_td_pred, newdata = drive_test, allow.new.levels = TRUE)
drive_test$td_prob <- exp(drive_test$pred_td)/(1+exp(drive_test$pred_td))

drive_test$pred_fg <- predict(drive_fg_pred, newdata = drive_test, allow.new.levels = TRUE)
drive_test$fg_prob <- exp(drive_test$pred_fg)/(1+exp(drive_test$pred_fg))

drive_test$pred_opp_td <- predict(drive_opp_td_pred, newdata = drive_test, allow.new.levels = TRUE)
drive_test$opp_td_prob <- exp(drive_test$pred_opp_td)/(1+exp(drive_test$pred_opp_td))

drive_test$pred_saf <- predict(drive_saf_pred, newdata = drive_test, allow.new.levels = TRUE)
drive_test$saf_prob <- exp(drive_test$pred_saf)/(1+exp(drive_test$pred_saf))

drive_test$pred_drive_pts <- 7*drive_test$td_prob + 3*drive_test$fg_prob -7*drive_test$opp_td_prob -2*drive_test$saf_prob

drive_test$resid <- drive_test$drive_points - drive_test$pred_drive_pts

mean(abs(drive_test$resid), na.rm = T)
```

The average residual is \~0.6 points. So on average, the model is off by a little more than half a point by a given drive. For a first crack at this problem with a non-ideal model, I think this is good enough to try and use it. But I am still unsure about this metric, and I am also unsure if it is an improvement over other metrics we have to evaluate drives. So let's run it through a ton of sniff tests and comparisons to see if it survives those.

## Comparisons and Sniff Tests

The first sniff test I want to do is I want to see what the distribution of predicted scores looks like for each type of score. What we expect to see is a little different for each type of score. On touchdown drives we expect to see a left-tailed distribution (most Touchdown drives are good, but there are some bad drives which score), for Field Goals we would expect to see it mostly centered around 3 PDP with a larger spread (good and bad drives can end in Field Goals), for Safeties we would expect a very large spike at roughly -2 PDP since there's a very narrow amount of drives which can reasonably end in a Safety, and for Opponent TDs we would expect a large spread since Pick 6s/Scoop 'n Scores can be pretty random.

So, let's check:

```{r}
test_tds <- drive_test %>% filter(touchdown == 1)
test_fgs <- drive_test %>% filter(fg == 1)
test_opp_tds <- drive_test %>% filter(opp_td == 1)
test_safs <- drive_test %>% filter(saf == 1)
```

This chunk just splits up the test set into subsets we care about.

```{r}
ggplot(data = test_tds)+
  geom_histogram(mapping = aes(x = pred_drive_pts), fill = 'red')+
  xlab('Drive Points')+
  ylab('Count')+
  labs(title = 'Distribution of Predicted Drive Points on Touchdown Drives')+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

This is pretty much exactly what we expected! We have a significant left tail, which suggests that some bad drives still end up as Touchdowns, but most of it is centered around 7. There is another issue where there's a significant amount of Touchdown drives which have *more* than 7 predicted points. This is a result of how we are actually calculating the drive points, since we are not using a multinomial logistic regression, sometimes the expected value will be greater than 7. So it is a flaw, but the fact that a lot of them are in the Touchdown distribution (a lot of the best drives end in Touchdowns) is a silver lining.

```{r}
ggplot(data = test_fgs)+
  geom_histogram(mapping = aes(x = pred_drive_pts), fill = 'red')+
  xlab('Drive Points')+
  ylab('Count')+
  labs(title = 'Distribution of Predicted Drive Points on Field Goals Drives')+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

This is not what we expected. For Field Goals is is actually centered at slightly less than 2.5 (eyeballing it, likely \~2) with a significant right tail. Is this concerning? A little bit. The model is worst at predicting Field Goals, which obviously an issue but this at least lets us diagnose the problem as under-predicting Field Goals. This certainly affects the accuracy of the model, but it is also possible that coaches are kicking Field Goals more often than they should on good drives, but that may just be me reading too deep into tea leaves. But most of the predicted points are still within 0.5 points of 2.5, so I don't feel great about this but I don't feel terrible about it either.

```{r}
ggplot(data = test_safs)+
  geom_histogram(mapping = aes(x = pred_drive_pts), fill = 'red')+
  xlab('Drive Points')+
  ylab('Count')+
  labs(title = 'Distribution of Predicted Drive Points on Safety Drives')+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

This isn't really what I expected at all. It is not at all centered about -2 predicted points, but at least nearly all of the drives have negative values, but there is a ton of variance with so few drives ending in safeties. I generally feel okay about this though, because it is likely that many drives ending in safeties had a few successful plays, but then a QB got sacked in the endzone or something similar to that. This would imply a non-zero Touchdown or Field Goal probability, which explains the greater than -2 predicted drive points. Also the AUC is extremely high, so I think this distribution is acceptable.

```{r}
ggplot(data = test_opp_tds)+
  geom_histogram(mapping = aes(x = pred_drive_pts), fill = 'red')+
  xlab('Drive Points')+
  ylab('Count')+
  labs(title = 'Distribution of Predicted Drive Points on Opponent Touchdown Drives')+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

Similar idea to what's going on with the safeties, but this makes more sense. This just means that Opponent TDs can kinda happen whenever, but most of the time they happen on bad drives. Or there are other indicators (notably, the total EPA of the drive) that are being caught up in the model. My guess is that it is the latter, and I am still not sure if that is necessarily an issue or not.

So I overall feel okay enough about the distributions of of each scoring drive that I think I *should* be able to use it. But first, I want to check whether or not I just recreated the cfbscrapR EP model or its derivatives. So we'll plot Predicted Drive Points against the drive's initial Expected Points, the drive's Total EPA, and the drive's EPA/Play. This won't necessarily mean that it is better at predicting outcomes than these metrics, it will just mean whether or not I just accidentally recreated them.

Since we've already established the model works generally, we'll apply it to the entire data set just to get as much data as possible to make sure I don't miss anything:

```{r}
all_drives$pred_td <- predict(drive_td_pred, newdata = all_drives, allow.new.levels = TRUE)
all_drives$td_prob <- exp(all_drives$pred_td)/(1+exp(all_drives$pred_td))

all_drives$pred_fg <- predict(drive_fg_pred, newdata = all_drives, allow.new.levels = TRUE)
all_drives$fg_prob <- exp(all_drives$pred_fg)/(1+exp(all_drives$pred_fg))

all_drives$pred_opp_td <- predict(drive_opp_td_pred, newdata = all_drives, allow.new.levels = TRUE)
all_drives$opp_td_prob <- exp(all_drives$pred_opp_td)/(1+exp(all_drives$pred_opp_td))

all_drives$pred_saf <- predict(drive_saf_pred, newdata = all_drives, allow.new.levels = TRUE)
all_drives$saf_prob <- exp(all_drives$pred_saf)/(1+exp(all_drives$pred_saf))

all_drives$pred_drive_pts <- 7*all_drives$td_prob + 3*all_drives$fg_prob -7*all_drives$opp_td_prob -2*all_drives$saf_prob

all_drives$resid <- all_drives$drive_points - all_drives$pred_drive_pts
```

Now we'll plot the Predicted Drive Points across each of the aforementioned metrics.

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = initial_ep, y = pred_drive_pts))+
  xlab("Initial EP")+
  ylab("Predicted Drive Points")+
  labs(title = "Predicted Drive Points vs Initial EP")+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

Quick interjection to note that this looks a lot like a [mermaid's purse](https://www.google.com/search?q=mermaid+purse&rlz=1C1CHBF_enUS894US894&sxsrf=ALeKk03VB0xPNoJQXwzCz9E4n1u3Mcgipw:1602643261948&tbm=isch&source=iu&ictx=1&fir=D1evbVZ74MsXiM%252CXHMT8_RVlgIs1M%252C_&vet=1&usg=AI4_-kTx9Aem2PyHY_ZSR8mfepS1LsfMcw&sa=X&ved=2ahUKEwiDt9Gyh7PsAhVJaM0KHbXGC3YQ9QF6BAgJEEs&biw=1536&bih=722#imgrc=D1evbVZ74MsXiM).

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = tepa, y = pred_drive_pts))+
  xlab("Total EPA")+
  ylab("Predicted Drive Points")+
  labs(title = "Predicted Drive Points vs Total EPA")+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = mean_epa, y = pred_drive_pts))+
  xlab("EPA/Play")+
  ylab("Predicted Drive Points")+
  labs(title = "Predicted Drive Points vs EPA/Play")+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

I think these two look like seahorses.

Full disclosure, I am not a Data Science or Statistics major, but I am almost certain that these are not meaningful direct relationships. Or at the very least, it is clear from here that PDP is not simply recreating Initial EP, Total EPA, or EPA/Play.

So now, let's see if the PDP model actually succeeds at predicting points on the drive level. If PDP successfully predicts points on the drive level, it should be able to predict points at the game level. If PDP successfully predicts points on the game level, it should be able to predict points on the season level. An important thing to note here is that this model will not necessarily predict Points Scored at the game or season level, but rather Total Points. The difference being if an offense throws a Pick-6, they score 0 points, so their Points Scored would remain the same while their Total Points would be -7 as they gave up 7 points. The two metrics essentially line up in most cases though, so it is a minor distinction.

```{r}
ggplot(data = all_drives)+
  geom_histogram(mapping = aes(x = drive_points), fill = 'red', alpha = 0.6)+
  geom_histogram(mapping = aes(x = pred_drive_pts), fill = 'blue', alpha = 0.6)+
  xlab("Drive Points")+
  ylab("Count")+
  labs(title = "Distribution of Drive Points and Predicted Drive Points",
       subtitle = "Red = Actual Drive Points, Blue = Predicted Drive Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_blank())
```

```{r}
ggplot(data = all_drives)+
  geom_histogram(mapping = aes(x = resid), fill = 'purple', alpha = 0.6)+
  xlab("Actual Drive Points - Predicted Drive Points")+
  ylab("Count")+
  labs(title = "Distribution of Residuals at Drive Level",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_blank())
```

The distributions of Predicted Drive Points and Drive-level points line up extremely well, aside from greatly under-predicting Field Goals which we already discussed. Additionally, the residuals are extremely small and centered at 0. So I feel safe to say that it does accurately predict Drive-level scoring, so let's see if it predicts game-level scoring.

```{r}
game_comp <- all_drives %>% 
  group_by(game_id, off) %>% 
  summarise(
    tot_points = sum(drive_points, na.rm = T), 
    tot_pred_points = sum(pred_drive_pts, na.rm = T),
    tepa = sum(tepa),
    num_plays = sum(num_plays),
    epa_play = tepa/num_plays,
    resid = tot_points - tot_pred_points
    ) 

ggplot(data = game_comp)+
  geom_histogram(mapping = aes(x = tot_points), fill = 'red', alpha = 0.6)+
  geom_histogram(mapping = aes(x = tot_pred_points), fill = 'blue', alpha = 0.6)+
  xlab("Game Total Points")+
  ylab("Count")+
  labs(title = "Distribution of Game-level Total Points",
       subtitle = "Red = Actual Total Points, Blue = Predicted Total Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.y = element_blank())
```

```{r}
ggplot(data = game_comp)+
  geom_histogram(mapping = aes(x = resid), fill = 'purple', alpha = 0.6)+
  xlab("Actual Total Points - Predicted Total Points")+
  ylab("Count")+
  labs(title = "Distribution of Game-level Total Points Residuals",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.y = element_blank())
```

Once again the distributions align, and the residuals are once again centered around 0. The residuals are a little higher, but that makes sense since drives individually have non-zero residuals so on the game level (\~10-12 drives usually) small residuals can pile up. Even with this, I feel comfortable in saying that it accurately predicts total points on the game level. So we will move on to analyzing the results at the season level. It is also important to note that most of the residuals are within 10 points.

```{r}
points_comp <- all_drives %>% 
  group_by(off, year) %>% 
  summarise(
    tot_points = sum(drive_points, na.rm = T),
    tot_pred_points = sum(pred_drive_pts, na.rm = T),
    resid = tot_points - tot_pred_points,
    tepa = sum(tepa, na.rm = T),
    n_plays = sum(num_plays),
    epa_play = tepa/n_plays
  ) %>% 
  arrange(desc(tot_pred_points)) %>% 
  filter(tot_points > 90)
#This filter is just to filter out FCS teams

ggplot(data = points_comp)+
  geom_histogram(mapping = aes(x = tot_points), fill = 'red', alpha = 0.6)+
  geom_histogram(mapping = aes(x = tot_pred_points), fill = 'blue', alpha = 0.6)+
  xlab("Season Total Points")+
  ylab("Count")+
  labs(title = "Distribution of Season Total Points and Season Predicted Total Points",
       subtitle = "Red = Actual Season Total Points, Blue = Predicted Season Total Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_blank())
```

```{r}
ggplot(data = points_comp)+
  geom_histogram(mapping = aes(x = resid), fill = 'purple', alpha = 0.6)+
  xlab("Actual Season Total Points - Predicted Season Total Points")+
  ylab("Count")+
  labs(title = "Distribution of Season Total Points Residuals",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_blank())
```

Even at the season-level, the distributions once again line up. The residuals once again are centered about 0 and roughly normal which is encouraging. The most encouraging bit is the upper and lower limits of the residuals which are 50 points. This means that even with the largest over performers and under performers, the model is only off by a little more than 4 points per game. I think this is really, really good.

One more sniff test to see whether or not the results actually make sense. We can look to see who the best and worst teams on the season level are according to the model, just as a sanity check.

```{r}
cfb_colors <- read.csv("https://raw.githubusercontent.com/mcqconor/PredictedDrivePoints/master/team_colors.csv")

points_comp$comp_name <-  paste((points_comp$year), points_comp$off, sep = " ")

colnames(cfb_colors)[1] <- "off"
points_comp <- points_comp %>% merge(cfb_colors)

best_20 <- points_comp %>% arrange(desc(tot_pred_points)) %>% head(20)
worst_20 <- points_comp %>% arrange((tot_pred_points)) %>% head(20)

ggplot(data = best_20)+
  geom_point(mapping = aes(y = tot_points, x = tot_pred_points), color = best_20$prim_color)+
  geom_text_repel(mapping = aes(y = tot_points, x = tot_pred_points, label = comp_name), family = 'Tahoma')+
  geom_text(mapping = aes(y = max(tot_points), x = min(tot_pred_points)+10, label = 'Overperformed', family = 'Tahoma'))+
  geom_text(mapping = aes(y = min(tot_points), x = max(tot_pred_points)-10, label = 'Underperformed', family = 'Tahoma'))+
  geom_abline(slope = 1, lty = 2, color = 'red')+
  xlab("Predicted Total Points")+
  ylab("Actual Total Points")+
  labs(title = "Top 20 Teams in Total Points in CFP Era",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

```{r}
ggplot(data = worst_20)+
  geom_point(mapping = aes(y = tot_points, x = tot_pred_points), color = best_20$prim_color)+
  geom_text_repel(mapping = aes(y = tot_points, x = tot_pred_points, label = comp_name), family = 'Tahoma')+
  geom_text(mapping = aes(y = max(tot_points), x = min(tot_pred_points)+10, label = 'Overperformed', family = 'Tahoma'))+
  geom_text(mapping = aes(y = min(tot_points), x = max(tot_pred_points)-10, label = 'Underperformed', family = 'Tahoma'))+
  geom_abline(slope = 1, lty = 2, color = 'red')+
  xlab("Predicted Total Points")+
  ylab("Actual Total Points")+
  labs(title = "Bottom 20 Teams in Total Points in CFP Era",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))
```

This checks the sanity check, as most of the top teams are Lincoln Riley Oklahomas, 2019 LSU, and various other playoff teams. LSU in my Twitter thread I posted was far and away the best offense ever, and this may be in part due to clipping off a game or two by looping over weeks 1-15 for both seasons. Additionally, between the time of posting the thread and present day there have been updates to cfbscrapR's EP model which may influence the discrepancy between these results. The results still remain that they make sense as to which teams are best and worst. The worst far and away is 2019 Akron.

So now we want to compare how predicted the PDP model is in comparison to other metrics. Notably, comparing PDP to drive points on the drive, game, and season level to the same for Total EPA and EPA/Play. This is to see whether this is better or worse than other metrics at what it is trying to do.

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = pred_drive_pts, y = drive_points))+
  geom_smooth(mapping = aes(x = pred_drive_pts, y = drive_points))+
  xlab("Predicted Drive Points")+
  ylab("Drive Points")+
  labs(title = "Drive Points vs Predicted Drive Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(all_drives$pred_drive_pts, all_drives$drive_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = mean_epa, y = drive_points))+
  geom_smooth(mapping = aes(x = mean_epa, y = drive_points))+
  xlab("EPA/Play")+
  ylab("Drive Points")+
  labs(title = "Drive Points vs EPA/Play",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(all_drives$mean_epa, all_drives$drive_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = all_drives)+
  geom_point(mapping = aes(x = tepa, y = drive_points))+
  geom_smooth(mapping = aes(x = tepa, y = drive_points))+
  xlab("Total EPA")+
  ylab("Drive Points")+
  labs(title = "Drive Points vs Total EPA",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(all_drives$tepa, all_drives$drive_points, use = "complete.obs")^2))
```

Graphically, it is difficult to ascertain whether PDP is better at predicting drive points than Total EPA or EPA/Play. Since this ought to be a linear relationship between any of the two variables (a higher PDP, Total EPA, or EPA/Play should directly correlate to more points) we should be able to measure how good each of the variables are by their R2 value. PDP has the highest R\^2 value of 0.84, then Total EPA with 0.71, and then EPA/Play with 0.40. I believe this is enough to assert that on the drive level, PDP is substantially better than Total EPA or EPA/Play at predicting drive points.

Now let's see what its like at the game level.

```{r}
ggplot(data = game_comp)+
  geom_point(mapping = aes(x = tot_pred_points, y = tot_points))+
  geom_smooth(mapping = aes(x = tot_pred_points, y = tot_points))+
  xlab("Total Predicted Game Points")+
  ylab("Total Game Points")+
  labs(title = "Total Game Points vs Total Predicted Game Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(game_comp$tot_pred_points, game_comp$tot_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = game_comp)+
  geom_point(mapping = aes(x = epa_play, y = tot_points))+
  geom_smooth(mapping = aes(x = epa_play, y = tot_points))+
  xlab("Game EPA/Play")+
  ylab("Total Game Points")+
  labs(title = "Total Game Points vs Game EPA/Play",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(game_comp$epa_play, game_comp$tot_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = game_comp)+
  geom_point(mapping = aes(x = tepa, y = tot_points))+
  geom_smooth(mapping = aes(x = tepa, y = tot_points))+
  xlab("Total Game EPA")+
  ylab("Total Game Points")+
  labs(title = "Total Game Points vs Total Game EPA",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(game_comp$tepa, game_comp$tot_points, use = "complete.obs")^2))
```

We repeat the same process on the game level, and the same logic. The difference here is that it is slightly easier to tell that PDP outperforms Total EPA and EPA/Play on the game level. To further quantify it, the R\^2s are highest for PDP with 0.90, next highest is Total EPA with 0.71, and the lowest is EPA/Play with 0.68. This still is strong evidence that PDP out performs other metrics in terms of total points on the game level.

Now let's repeat this one more time on the season level:

```{r}
ggplot(data = points_comp)+
  geom_point(mapping = aes(x = tot_pred_points, y = tot_points))+
  geom_smooth(mapping = aes(x = tot_pred_points, y = tot_points))+
  xlab("Total Predicted Season Points")+
  ylab("Total Season Points")+
  labs(title = "Total Seaon Points vs Total Predicted Season Points",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(points_comp$tot_pred_points, points_comp$tot_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = points_comp)+
  geom_point(mapping = aes(x = epa_play, y = tot_points))+
  geom_smooth(mapping = aes(x = epa_play, y = tot_points))+
  xlab("Season EPA/Play")+
  ylab("Total Season Points")+
  labs(title = "Total Seaon Points vs Season EPA/Play",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(points_comp$epa_play, points_comp$tot_points, use = "complete.obs")^2))
```

```{r}
ggplot(data = points_comp)+
  geom_point(mapping = aes(x = tepa, y = tot_points))+
  geom_smooth(mapping = aes(x = tepa, y = tot_points))+
  xlab("Season Total EPA")+
  ylab("Total Season Points")+
  labs(title = "Total Seaon Points vs Season Total EPA",
       caption = "Data from @cfbscrapR and @CFB_Data | By: Conor McQuiston @ConorMcQ5")+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.caption = element_text(family = "Tahoma"))+
  theme(panel.background = element_rect(fill = "seashell", color = "gray", size = 0.5, linetype = "solid"))+
  theme(plot.background = element_rect(fill = "seashell"))+
  theme(panel.grid.major.y = element_blank(),
        panel.grid.major.x = element_blank())+
  theme(panel.grid.minor=element_blank())+
  theme(plot.title = element_text(hjust = 0.5, family = "Tahoma", size = 15),
        plot.subtitle = element_text(hjust = 0.5, family = "Tahoma", size = 13),
        plot.caption = element_text(family = "Tahoma"))+
  theme(axis.title.y = element_text(size = 13, family = "Tahoma"),
        axis.title.x = element_text(size = 13, family = "Tahoma"),
        axis.text.x = element_text(size = 11, family = "Tahoma"),
        axis.text.y = element_text(size = 11, family = "Tahoma"))

print(toString(cor(points_comp$tepa, points_comp$tot_points, use = "complete.obs")^2))
```

Yet again on the season level, it is graphically apparent that PDP is much better correlated to Total Points on the season level than Total EPA and EPA/Play. If you do not agree, we can quantify this via its R\^2 values, PDP has the highest with 0.96, Total EPA is next with 0.83, and EPA/Play is the least with 0.81. Although all 3 values are all very high, PDP represents a significant improvement over Total EPA and EPA/Play.

## Summary

My Predicted Drive Points (PDP) model is a series of logistic regressions which is meant to evaluate the quality of a drive independent of the drive's outcome. Each logistic regression passes the requisite AUC test, and the model overall has a mean absolute residual of 0.57 points. It appears that the model is more than a simple recreation of a drive's initial EPA, Total EPA, or EPA/Play and it passes all "sniff tests" performed. These sniff tests involved matching the distributions of PDP and Actual Total Points at the drive, game, and season level and the respective residuals and looking at the best and worst teams according to the model. Additionally, R\^2s were compared between PDP, Total EPA, and EPA/Play and Actual Total Points at the drive, game, and season levels in order to determine that PDP was a significant improvement in predicting points at each level over the other metrics.

Thanks for reading! This was really long! If you enjoyed this and want to see some applications of this metric, follow me on Twitter [\@ConorMcQ5](https://twitter.com/ConorMcQ5).
