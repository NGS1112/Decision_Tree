% Filename:    Decision_Tree_main.m
% Author:      Nicholas Shinn
% Class:       Principles of Data Mining
% Description: Program that writes a decision tree from a given csv training file.
%              Performs recursive binary splitting to minimize weighted entropy.


function Decision_Tree_main(filename)
% Decision_Tree_main - Mentor program for exercise 05 of PDM
%   filename - Name of training file to be used for tree construction

    % Constant global variables used for stopping criteria
    global maxTreeDepth minNodePurity minDataNodes
    maxTreeDepth = 10;
    minNodePurity = 95/100;
    minDataNodes = 9;

    %Opens the initial data being read in from file
    training_data = readmatrix(filename);

    %Quantizes the data to the appropriate units (bins of 2 for all
    %variables except height which is binned at 4)
    for column = 1:6
        if( column ~= 2 )
            training_data( :, column ) = round(training_data( :, column) / 2) * 2;
        else
            training_data( :, column ) = round(training_data( :, column) / 4) * 4;
        end
    end

    %Attempts to create the trained program file, begins execution if successful
    answer = fopen('out/Decision_Tree_classifier.m', 'wt');
    if answer ~= -1
        %Initial header to print in the trained program
        fprintf(answer, 'function Decision_Tree_classifier(filename)\n');
        fprintf(answer, "records = fopen('out/Classifications.csv', 'wt');\n");
        fprintf(answer, 'data = readmatrix(filename);\n');
        fprintf(answer, 'rows = size(data, 1);\n');
        fprintf(answer, 'for row = 1:rows\n');
    
        %Begins the recursive calls of the decision tree maker
        Binary_Splitter(training_data, 0, 0, answer);
    
        %Footer for once the tree has been built
        fprintf(answer, 'end\n');
        fprintf(answer, 'fclose(records);\n');
        fprintf(answer, 'end');
    end
    
    %Wraps things up by closing the created class
    fclose(answer);
end

function Binary_Splitter(dataset, depth, fallback, writeToFile)
%Binary_Splitter - Recursive function that performs binary splits to build
%                  a decision tree
%   dataset    - The current set of entries being analyzed
%   depth      - The current depth the program is at in the tree
%   fallback   - Majority classification of parent node
%   writeToFile- File to write classifications into

    %Grabs the global stopping values previously set
    global maxTreeDepth minNodePurity minDataNodes
    
    %Grabs the current classifications of entries in the dataset
    classified_assam = sum(dataset(:, 9) == +1);
    classified_bhutan = sum(dataset(:, 9) == -1);
    total = classified_assam + classified_bhutan;
    
    %Sets the majority classification and associated symbols based on
    %majority class of current dataset or from parent class if there is a
    %tie
    if( classified_assam > classified_bhutan )
        dominant_class = +1;
        class_marker = "Assam";
        class_symbol = '+';
    elseif( classified_bhutan > classified_assam )
        dominant_class = -1;
        class_marker = "Bhutan";
        class_symbol = '-';
    else
        dominant_class = fallback;
        if( dominant_class == +1 )
            class_marker = "Assam";
            class_symbol = '+';
        else
            class_marker = "Bhutan";
            class_symbol = '-';
        end
    end
    
    %If we have reached a stopping condition, print the appropriate lines and exit function call
    if( dominant_class/total > minNodePurity || total <= minDataNodes || depth >= maxTreeDepth )
        Tab_Builder(depth, writeToFile);
        fprintf(writeToFile, "fprintf('class = %c1 for %s\\n');\n", class_symbol, class_marker);
        Tab_Builder(depth, writeToFile);
        fprintf(writeToFile, "fprintf(records, '%c1\\n');\n", class_symbol);
        return
    end
    
    %Sets fallback to current majority classification for next recursive
    %call
    if( classified_assam > classified_bhutan )
        fallback = +1;
    elseif( classified_bhutan > classified_assam )
        fallback = -1;
    end
    
    %Initializes tracker variables to be used for feature/threshold
    %selection
    best_feature = Inf;
    best_threshold = Inf;
    best_entropy = Inf;
    best_split = Inf;
    
    %Iterates over each possible feature and threshold, selecting the one
    %that produces the minimal weighted entropy
    for feature = 1:6
        for threshold = unique( dataset( :, feature ) )'
            
            %Splits the dataset into left and right subsets based on
            %threshold
            left = dataset( dataset(:, feature) >= threshold, :);
            right = dataset( dataset(:, feature) < threshold, : );
            
            %Calculates the probability of an entry in the left subset
            %being assam or bhuttan
            left_assam = sum(left(:, 9) == +1.0);
            left_bhutan = sum(left(:, 9) == -1.0);
            total_left = left_assam + left_bhutan;
            p_left_assam = left_assam / total_left;
            p_left_bhutan = left_bhutan / total_left;
            
            %If entropy would produce a non-number, set it to zero.
            %Otherwise, calculate entropy values for each possible class on
            %the left subset
            if( p_left_assam == 0 )
                entropy_left_assam = 0;
            else
                entropy_left_assam = -(p_left_assam) * log2(p_left_assam);
            end
            if( p_left_bhutan == 0 )
                entropy_left_bhutan = 0;
            else
                entropy_left_bhutan = -(p_left_bhutan) * log2(p_left_bhutan);
            end
            
            %Combine the previously calculated entropy values together for
            %the left subset
            entropy_left = entropy_left_assam + entropy_left_bhutan;
            
            %Calculates the probability of an entry in the right subset
            %being assam or bhuttan
            right_assam = sum(right(:, 9) == +1.0);
            right_bhutan = sum(right(:, 9) == -1.0);
            total_right = right_assam + right_bhutan;
            p_right_assam = right_assam / total_right;
            p_right_bhutan = right_bhutan / total_right;
            
            %If entropy would produce a non-number, set it to zero.
            %Otherwise, calculate entropy values for each possible class on
            %the right subset
            if( p_right_assam == 0 )
                entropy_right_assam = 0;
            else
                entropy_right_assam = -(p_right_assam) * log2(p_right_assam);
            end
            if( p_right_bhutan == 0 )
                entropy_right_bhutan = 0;
            else
                entropy_right_bhutan = -(p_right_bhutan) * log2(p_right_bhutan);
            end
            
            %Combine the previously calculated entropy values together for
            %the right subset
            entropy_right = entropy_right_assam + entropy_right_bhutan;
            
            %Calculates the weights for the left and right subsets before
            %calculating the weighted entropy
            total = total_left + total_right;
            weighted_entropy = (total_left / total) * entropy_left + (total_right / total) * entropy_right;
            
            %Calculate difference in size of left and right halves to be
            %used in event of a tie
            split = abs(total_left - total_right);
            
            %If the calculated weighted entropy is less than our current
            %best, set this split as the best split
            if( weighted_entropy < best_entropy )
                best_feature = feature;
                best_threshold = threshold;
                best_entropy = weighted_entropy;
                best_split = split;
            elseif( weighted_entropy == best_entropy && split > best_split )
                best_feature = feature;
                best_threshold = threshold;
                best_entropy = weighted_entropy;
                best_split = split;
            end
        end
    end

    
    %Recursively builds the tree by performing a binary split on the best
    %feature and threshold, calling this function with updated datasets and
    %tree depths
    Tab_Builder(depth, writeToFile);
    fprintf(writeToFile, 'if( data(row, %i) >= %i )\n', best_feature, best_threshold);
    Binary_Splitter( dataset( dataset(:, best_feature) >= best_threshold, :), depth + 1, fallback, writeToFile );
    Tab_Builder(depth, writeToFile);
    fprintf(writeToFile, 'else\n');
    Binary_Splitter( dataset( dataset(:, best_feature) < best_threshold, :), depth + 1, fallback, writeToFile );
    Tab_Builder(depth, writeToFile);
    fprintf(writeToFile, 'end\n');
end

function Tab_Builder(treeDepth, writeToFile)
%Tab_Builder - Inserts appropriate amount of tabs needed into file
%   treeDepth    - Current depth the program is at in the decision tree
%   writeToFile  - File to write the tabs into

    %Inserts one tab character for each level we have passed in the tree
    for count = 0:treeDepth
        fprintf(writeToFile, '\t');
    end
end
